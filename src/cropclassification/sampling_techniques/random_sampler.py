import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio


def randon_sampling(num_points:int, output_dir:str, buffer_size:float, crop_types_path:str, output_name:str)-> None:
    """
    Generate random points within a buffer around the edges of a raster, assigning each point a class ID based on the raster values.

    Parameters:
    - num_points (int): Number of random points to generate.
    - buffer_size (float): Size of the buffer around the edges of the raster.
    - crop_types_path (str): Path to the raster file containing crop types.
    - output_shapefile (str): Path to the output shapefile to save the generated points.

    Returns:
    - None
    """
    samples_points_dir = os.path.join(output_dir, 'results', 'sample_points', 'random_sampling')
    # Create the output directory if it doesn't exist
    os.makedirs(samples_points_dir, exist_ok=True)
    # Generate random points with a buffer around the edges and assign class ID
    output_shapefile = os.path.join(samples_points_dir, output_name)
    # Open the crop_types raster to get its bounding box
    with rasterio.open(crop_types_path) as raster:
        raster_bounds = raster.bounds

    xmin, ymin, xmax, ymax = raster_bounds
    inner_xmin, inner_ymin, inner_xmax, inner_ymax = xmin + buffer_size, ymin + buffer_size, xmax - buffer_size, ymax - buffer_size
    random_coordinates = np.column_stack((np.random.uniform(inner_xmin, inner_xmax, num_points),
                                          np.random.uniform(inner_ymin, inner_ymax, num_points)))

    # Open the crop_types raster to get the class IDs
    with rasterio.open(crop_types_path) as crop_raster:
        crop_type_values = list(crop_raster.sample(random_coordinates))

    # Create a GeoDataFrame for the random points with class ID
    gdf = gpd.GeoDataFrame({'geometry': [Point(coord) for coord in random_coordinates],
                            'class_id': [value[0] for value in crop_type_values]},
                           crs=crop_raster.crs)

    # Export the GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile)





