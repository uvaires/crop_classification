import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import xy
import pandas as pd
import rasterio


def test_dataset(image_path: str, base_dir: str, samples_per_class: dict, buffer_size=10,
                        output_name=None) -> None:
    """
    Generates random points within a specified buffer around the edges of each class in a raster image and exports them to shapefiles.

    Parameters:
    - image_path (str): Path to the raster image file.
    - base_dir (str): Base directory where the output shapefiles will be saved.
    - samples_per_class (dict): Dictionary specifying the number of samples to generate for each class.
    - buffer_size (int, optional): Size of the buffer around the edges of each class. Default is 10.
    - output_name (str, optional): Name of the output shapefile. If None, a default name will be used.
    """
    samples_points_dir = os.path.join(base_dir, 'results', 'test_dataset')
    # Create the output directory if it doesn't exist
    os.makedirs(samples_points_dir, exist_ok=True)

    # Set default output name if not provided
    if output_name is None:
        output_name = 'stratified_samples.shp'

    # Export the GeoDataFrame as a shapefile
    output_shapefile = os.path.join(samples_points_dir, output_name)
    class_count = {}
    gdf_list = []  # Initialize an empty list to store GeoDataFrames

    with rasterio.open(image_path) as src:
        raster_data = src.read(1)
        transform = src.transform
        crs = src.crs

        for class_value, target_samples in samples_per_class.items():
            if class_value >= 0:  # Ensure non-negative class values
                indices = np.where(raster_data == class_value)

                coordinates = [xy(transform, i, j) for i, j in zip(indices[0], indices[1]) if
                               i > buffer_size and i < src.width - buffer_size and j > buffer_size and j < src.height - buffer_size]

                # Sample points randomly
                sampled_indices = np.random.choice(
                    len(coordinates), min(target_samples, len(coordinates)), replace=False
                )
                sampled_coordinates = [coordinates[i] for i in sampled_indices]

                class_count[class_value] = len(sampled_coordinates)

                value_list = [class_value] * len(sampled_coordinates)

                gdf = gpd.GeoDataFrame(  # Create a GeoDataFrame for each class
                    {'geometry': [Point(coord) for coord in sampled_coordinates], 'value': value_list}, crs=crs
                )
                gdf_list.append(gdf)  # Append the GeoDataFrame to the list

        # Merge all GeoDataFrames into one
        final_gdf = pd.concat(gdf_list, ignore_index=True)
        # Export to shapefile
        final_gdf.to_file(output_shapefile)