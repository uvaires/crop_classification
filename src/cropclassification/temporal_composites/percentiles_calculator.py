import glob
import os
import rasterio
from collections import defaultdict
import numpy as np
from datetime import datetime
import fastnanquantile as fnq

def calculate_percentiles_hls(base_dir:str, date_ranges:dict, bands:list, percentiles = [0.10, 0.25, 0.50, 0.75, 0.90])->None:
    """
    Process images, calculate percentiles, and export images based on specified date ranges.

    Args:
        base_dir (str): read the cloudless HLS images and save percentile images.
        img_dict (dict): Dictionary associating image types with their file paths.
        date_ranges (dict): Dictionary containing date ranges for each band.

    Returns:
        None
    """
    # Create a dictionary to store the monthly percentiles for each band and period
    monthly_percentiles = defaultdict(list)

    img_dict = {}
    for band in bands:
        img_dict[band] = f'**/hls_cloudless/**/*{band}.tif'

    for band, path_pattern in img_dict.items():
        for date_range_key, (start_date, end_date) in date_ranges.items():
            if band in date_range_key:
                img_paths = glob.glob(os.path.join(base_dir, path_pattern), recursive=True)
                img_layers, img_profile, product, image_dates = _load_img_layers(img_paths)

                # Convert image dates to datetime objects with only month and year
                datetime_dates = [datetime.strptime(date.split('_')[0], "%Y%m%d") for date in image_dates]

                # Calculate percentiles for the current date range
                start_index = None
                end_index = None

                for i, date in enumerate(datetime_dates):
                    if date >= start_date and (end_index is None or date <= end_date):
                        if start_index is None:
                            start_index = i
                        end_index = i

                if start_index is not None and end_index is not None:
                    current_month = datetime_dates[start_index].month
                    selected_images = img_layers[start_index:end_index + 1]

                    # Calculate percentiles
                    percentiles = percentiles
                    percentile_imgs = fnq.nanquantile(selected_images, percentiles, axis=0)

                    # Ensure that there is a list for the current band in monthly_percentiles
                    if band not in monthly_percentiles:
                        monthly_percentiles[band] = []

                    # Append the percentiles to the corresponding band in monthly_percentiles
                    for percentile, img_data in zip(percentiles, percentile_imgs):
                        monthly_percentiles[band].append((current_month, percentile, img_data))

    # Call export_img with monthly_percentiles
    _export_img(monthly_percentiles, base_dir, img_profile)



#### Private functions #####
def _read_image(image_path):
    """
    Reads a raster image from the given file path using rasterio.

    Args:
        image_path (str): The file path to the raster image.

    Returns:
        Tuple: Tuple containing the image layer and its profile.
    """
    with rasterio.open(image_path) as src:
        img_layer = src.read(1)
        img_profile = src.profile
    return img_layer, img_profile


def _load_img_layers(img_paths):
    """
    Loads multiple image layers from the given list of image paths.

    Args:
        img_paths (list): List of file paths to the raster images.

    Returns:
        Tuple: Tuple containing a list of image layers, band profile, product names, and image dates.
    """
    img_layers = []
    band_profile = None
    product = []
    image_dates = []

    for img_path in img_paths:
        print(img_path)
        product_hls = os.path.basename(img_path).split('_')[1]
        product.append(product_hls)
        img_layer, band_profile = _read_image(img_path)
        img_layers.append(img_layer)
        dates = os.path.basename(img_path).split('_')[0]
        image_dates.append(dates)

    img_layers = np.stack(img_layers)
    return img_layers, band_profile, product, image_dates

def _export_img(percentiles, output_dir, img_profile):
    """
    Exports percentile images based on the provided monthly percentiles.

    Args:
        percentiles (dict): Dictionary containing percentiles for each band.
        output_dir (str): Output directory for saving percentile images.
        img_profile (dict): Image profile used for exporting.

    Returns:
        None
    """
    # Mapping quantiles to their corresponding labels
    quantile_labels = {
        0.1: "10",
        0.25: "25",
        0.5: "50",
        0.75: "75",
        0.9: "90"
    }

    for band, percentile_data in percentiles.items():
        for current_month, percentile, img_data in percentile_data:
            current_date = f"{current_month:02d}"  # Assuming zero-padded months
            img_names = band

            dir_output_date = os.path.join(output_dir, 'temporal_composites', 'percentiles')
            os.makedirs(dir_output_date, exist_ok=True)

            # Get the label for the current quantile
            quantile_label = quantile_labels[percentile]

            # Export percentile image
            output_filepath_percentile = os.path.join(dir_output_date, f"{current_date}_{img_names}_p{quantile_label}.tif")
            img_profile['nodata'] = np.nan
            img_profile['compress'] = 'lzw'

            with rasterio.open(output_filepath_percentile, 'w', **img_profile) as dst:
                dst.write(img_data, 1)
                print(f"Saved {quantile_label}% percentile image: {output_filepath_percentile}")


