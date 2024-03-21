import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import xy
import pandas as pd
import rasterio


def gridded_sampling(output_dir, image_path, total_points, output_name, buffer_size=10):

    # Create a folder to store the sample points
    samples_points_dir = os.path.join(output_dir, 'results', 'sample_points', 'gridded_sampling')
    os.makedirs(samples_points_dir, exist_ok=True)

    # Full path for the output shapefile
    output_shapefile = os.path.join(samples_points_dir, os.path.basename(output_name))

    gdf_list = []

    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs

        # Adjust the grid size to achieve the desired total number of points
        grid_size = int(np.sqrt(total_points)) + 1

        # Generate regular grid points covering the entire raster
        xmin, ymin = xy(transform, buffer_size, buffer_size)
        xmax, ymax = xy(transform, src.width - buffer_size, src.height - buffer_size)

        x = np.linspace(xmin, xmax, grid_size)
        y = np.linspace(ymin, ymax, grid_size)

        grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        # Trim the excess points to meet the target total
        grid_points = grid_points[:total_points]

        # Read raster values for each grid point
        values = list(src.sample(grid_points))
        value_list = [val[0] for val in values]

        gdf_list.append(
            gpd.GeoDataFrame(
                {'geometry': [Point(coord) for coord in grid_points], 'value': value_list}, crs=crs
            )
        )

    # Export the sample points
    gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    gdf.to_file(output_shapefile)




