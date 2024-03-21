import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import xy
import pandas as pd
import rasterio
from sklearn.cluster import DBSCAN


def clustered_sampling(crop_types, base_dir, samples_per_class, min_pixels, eps, min_samples, min_cluster_size, min_density, output):
    """
    Generate clustered points from a raster image for specified classes.

    Parameters:
        crop_types (str): Path to the raster image.
        base_dir (str): Base directory for saving sample points.
        samples_per_class (dict): Dictionary specifying the number of samples per class.
        min_pixels (int): Minimum number of pixels of the same value to consider a class.
        eps (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter).
        min_samples (int): Number of samples (or total weight) in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
        min_cluster_size (int): Number of samples in a cluster for it to be considered significant (DBSCAN parameter).
        min_density (float): Minimum density required for a cluster to be considered (density = cluster_size / total_samples).
        output_shapefile (str): Path to the output shapefile.

    Returns:
        gdf (GeoDataFrame): GeoDataFrame containing clustered points for all classes.
    """
    class_count = {}
    all_sampled_points = []

    # Create folder structure for storing sample points
    samples_dir = os.path.join(base_dir, 'results', 'sample_points', 'clustered_sampling')
    os.makedirs(samples_dir, exist_ok=True)
    sample_points = os.path.join(samples_dir, output)

    with rasterio.open(crop_types) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs

        for class_value, target_samples in samples_per_class.items():
            if class_value >= 0:
                indices = np.where(raster_data == class_value)

                if len(indices[0]) >= min_pixels:
                    buffer_size = 10
                    coordinates = [xy(transform, i, j) for i, j in zip(indices[0], indices[1]) if
                                   buffer_size < i < src.width - buffer_size and buffer_size < j < src.height - buffer_size]

                    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
                    unique_labels, cluster_sizes = np.unique(clustering.labels_, return_counts=True)

                    for label, size in zip(unique_labels, cluster_sizes):
                        if size >= min_cluster_size:
                            cluster_indices = np.where(clustering.labels_ == label)[0]
                            cluster_density = size / len(coordinates)

                            if cluster_density >= min_density:
                                sampled_indices = np.random.choice(
                                    len(cluster_indices), min(target_samples, len(cluster_indices)), replace=False
                                )
                                sampled_coordinates = [coordinates[i] for i in cluster_indices[sampled_indices]]

                                class_count[class_value] = len(sampled_coordinates)
                                value_list = [class_value] * len(sampled_coordinates)

                                all_sampled_points.extend(
                                    [{'geometry': Point(coord), 'value': class_value} for coord in sampled_coordinates]
                                )

    gdf = gpd.GeoDataFrame(all_sampled_points, crs=crs)
    gdf.to_file(sample_points)

    return gdf


