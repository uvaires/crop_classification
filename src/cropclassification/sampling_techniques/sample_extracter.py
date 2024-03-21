import geopandas as gpd
import rasterio
import os
import glob
import pandas as pd
import numpy as np

def extract_training_samples(base_dir: str, samples_points_dir: str) -> None:
    """
    Process raster data for each shapefile point and save the results to Excel files.

    Parameters:
    - base_dir (str): Base directory for input data.
    - samples_points_dir (str): Directory containing the shapefile points.

    Returns:
    - None
    """
    # Create folder to storage the training samples
    output_dir = os.path.join(base_dir, 'results', 'training_samples')
    os.makedirs(output_dir, exist_ok=True)

    def extract_raster_values_to_dataframe(img_paths, gdf):
        result_df = gdf.copy()

        for raster_path in img_paths:
            print(raster_path)
            with rasterio.open(raster_path) as src:
                values_list = []

                for index, row in gdf.iterrows():
                    # Get the geometry of the point
                    point_geometry = row['geometry']

                    # Extract the values from the raster for the point
                    try:
                        values = list(src.sample([point_geometry.coords[0]]))
                        values_list.append(values)
                    except Exception as e:
                        print(f"Error extracting values for Point {index} from {raster_path} - Error: {e}")

                # Create a DataFrame with the values for the current raster layer
                raster_df = pd.DataFrame(values_list,
                                         columns=[f"Raster_{raster_path}_{i}" for i in range(len(values_list[0]))])

                # Concatenate the new DataFrame to the result_df
                result_df = pd.concat([result_df, raster_df], axis=1)

        return result_df

    def extract_date_from_layer(layer_path):
        names = []
        for layer_paths in layer_path:
            # Extract the file name without extension
            file_name = os.path.splitext(os.path.basename(layer_paths))[0]
            names.append(file_name)

        return names

    # Locate the images
    img_paths = glob.glob(os.path.join(base_dir,'**' ,'inputdata', '**', '*.tif'), recursive=True)

    # Iterate over all shapefiles in the directory
    for shapefile_path in glob.glob(os.path.join(samples_points_dir,'**' , '*.shp'), recursive=True):
        # Read shapefile points
        gdf = gpd.read_file(shapefile_path)

        # Apply function to extract the raster values to point
        result_df = extract_raster_values_to_dataframe(img_paths, gdf)

        # Convert the GeoDataFrame to a regular DataFrame
        result_df = result_df.drop(columns='geometry')

        # Apply a lambda function to convert values to regular numbers
        result_df = result_df.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

        # Rename columns
        layer_date = extract_date_from_layer(img_paths)

        layer_date.insert(0, 'class')

        # Check if the combined_list has enough elements to rename columns
        if len(layer_date) == result_df.shape[1]:
            # Rename columns using 'combined_list'
            result_df.columns = layer_date

        # Move the first column to the last position
        columns = list(result_df.columns)
        columns.append(columns.pop(0))
        result_df = result_df[columns]

        # Define the output Excel file path including the shapefile name
        output_excel_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(shapefile_path))[0]}.xlsx")

        # Export the DataFrame to an Excel file
        result_df.to_excel(output_excel_path, index=False)
















