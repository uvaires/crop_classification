import glob
import os
import rasterio
import numpy as np
from datetime import datetime


def process_hls_images(base_dir: str, scaler_factor=10000) -> None:
    """
    Process HLS images: remove clouds, select bands of interest and rename NIR, SWR1, and SWR2

    Args:

        base_dir (str): Output directory where processed images are stored.

    """
    img_folders = glob.glob(os.path.join(base_dir, '**', 'hls_organized', '**', '*.tif'), recursive=True)
    bands_per_product = {
        'S30': {'B02', 'B03', 'B04', 'NIR', 'SWR1', 'SWR2'},
        'L30': {'B02', 'B03', 'B04', 'B05', 'SWR1', 'SWR2'}
    }

    for hls_images in img_folders:
        # Extract information from the HLS image filename
        img_dates = os.path.basename(hls_images).split('_')[0]
        img_product = os.path.basename(hls_images).split('_')[1].split('_')[0]
        tile = os.path.basename(hls_images).split('_')[2]

        # Create a directory to store the cloudless images
        dir_output_date = os.path.join(base_dir, 'pre_process', 'hls_cloudless', img_dates)
        os.makedirs(dir_output_date, exist_ok=True)

        # Load the cloud mask for the current image
        cloud_mask_images = os.path.join(base_dir, 'pre_process', 'fmask_decoded', img_dates,
                                         f'{img_dates}_{img_product}_{tile}_Fmask.tif')

        with rasterio.open(cloud_mask_images) as src:
            cloud_mask = src.read(1)

        bands_to_process = bands_per_product.get(img_product, set())
        for band_name in bands_to_process:
            # Find the path to the original band
            file_band = os.path.join(base_dir, 'pre_process', 'hls_organized', img_dates,
                                     f'{img_dates}_{img_product}_{tile}_{band_name}.tif')
            if not os.path.exists(file_band):
                continue  # Skip if the band file doesn't exist

            # Load the band
            with rasterio.open(file_band) as band_src:
                band_data = band_src.read(1) / float(scaler_factor)
                src_profile = band_src.profile

            # Replace values outside the range [0, 1] with np.nan
            band_data[(band_data > 1) | (band_data < 0)] = np.nan

            # Apply the cloud mask to the band
            band_data = np.where(cloud_mask == 1, np.nan, band_data)

            output_path = os.path.join(dir_output_date, os.path.basename(file_band))
            src_profile['nodata'] = np.nan
            src_profile['dtype'] = 'float32'
            src_profile['compress'] = 'lzw'
            with rasterio.open(output_path, 'w', **src_profile) as dst:
                dst.write(band_data.astype(rasterio.float32), 1)




