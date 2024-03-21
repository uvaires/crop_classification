import glob
import os
import rasterio
from collections import defaultdict
import numpy as np
from datetime import datetime

def calculate_hls_cv(base_dir: str, date_ranges: dict, bands:dict) -> None:
    """
    Processes images based on date ranges and exports quarterly variation of coefficient (cv) images.

    Args:
    - base_dir (str): Locate the hls cloudless images and save the coefficient of variation images
    - date_ranges (dict): Dictionary mapping band-date ranges to start and end dates
    - bands (dict): List of interes bands
    """
    monthly_cv = defaultdict(list)
    img_dict = {}
    for band in bands:
        img_dict[band] = f'**/hls_cloudless/**/*{band}.tif'


    for band, path_pattern in img_dict.items():
        for date_range_key, (start_date, end_date) in date_ranges.items():
            if band in date_range_key:
                img_paths = glob.glob(os.path.join(base_dir, path_pattern), recursive=True)
                img_layers, img_profile, product, image_dates = _load_img_layers(img_paths)

                datetime_dates = [datetime.strptime(date.split('_')[0], "%Y%m%d") for date in image_dates]

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

                    cv_img = np.nanstd(selected_images, axis=0) / np.nanmean(selected_images, axis=0)

                    if band not in monthly_cv:
                        monthly_cv[band] = []

                    monthly_cv[band].append((current_month, cv_img))

    _export_img(monthly_cv, base_dir, img_profile)


### Private functions ###

def _read_image(image_path):
    """
    Reads a raster image using rasterio and returns the image layer and profile.

    Args:
    - image_path (str): Path to the raster image.

    Returns:
    - img_layer (numpy.ndarray): Image layer as a NumPy array.
    - img_profile (rasterio.profiles.Profile): Image profile containing metadata.
    """
    with rasterio.open(image_path) as src:
        img_layer = src.read(1)
        img_profile = src.profile
    return img_layer, img_profile

def _load_img_layers(img_paths):
    """
    Loads image layers from a list of image paths.

    Args:
    - img_paths (list of str): List of paths to raster images.

    Returns:
    - img_layers (list of numpy.ndarray): List of image layers.
    - band_profile (rasterio.profiles.Profile): Profile of the last loaded image.
    - product (list of str): List of product names extracted from image paths.
    - image_dates (list of str): List of image dates extracted from image paths.
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

    return img_layers, band_profile, product, image_dates

def _export_img(monthly_cv, output_dir, img_profile):
    """
    Exports coefficient of variation images to the specified output directory.

    Args:
    - monthly_cv (default-dict): Dictionary containing monthly coefficient of variation for each band.
    - output_dir (str): Output directory for saving images.
    - img_profile (rasterio.profiles.Profile): Profile used for exporting images.
    """
    for band, monthly_data in monthly_cv.items():
        for current_month, cv_img in monthly_data:
            current_date = f"{current_month:02d}"
            img_names = band

            dir_output_date = os.path.join(output_dir, 'temporal_composites', 'inputdata', 'variation_coefficient')
            os.makedirs(dir_output_date, exist_ok=True)

            # Export CV image
            output_filepath_cv = os.path.join(dir_output_date, f"{current_date}_{img_names}_cv.tif")
            img_profile['nodata'] = np.nan
            img_profile['compress'] = 'lzw'

            with rasterio.open(output_filepath_cv, 'w', **img_profile) as dst:
                dst.write(cv_img, 1)
                print(f"Saved CV image: {output_filepath_cv}")


