import numpy as np
import rasterio
import glob
import os


def process_fmask(base_dir: str) -> None:
    """
    Process Fmask images in the specified input directory.

    Args:
        output_dir (str): Output directory for the processed images.
        hls_folder (str): Subdirectory where Fmask images are located within the input directory.
    """
    # Step 1: Get a list of paths for Fmask images in the specified input directory
    img_fmask = glob.glob(os.path.join(base_dir, '**', 'hls_organized', '**', '*Fmask.tif'), recursive=True)

    # Step 2: Loop through each Fmask image and decode it
    for hls_fmask in img_fmask:
        # Extract date, product type, and tile information from the file name
        img_dates = os.path.basename(hls_fmask).split('_')[0]
        img_product = os.path.basename(hls_fmask).split('_')[1].split('_')[0]
        tile = os.path.basename(hls_fmask).split('_')[2]

        # Step 3: Create a directory to store the processed images based on the image date
        dir_output_date = os.path.join(base_dir, 'pre_process', 'fmask_decoded', img_dates)
        os.makedirs(dir_output_date, exist_ok=True)
        print(hls_fmask)

        # Step 4: Open the Fmask image
        with rasterio.open(hls_fmask) as src:
            img_arr = src.read(1)
            profile = src.profile.copy()  # Copy the profile of the source image

        # Step 5: Update the profile for the output image
        profile.update(count=1, dtype=rasterio.uint8)

        # Step 6: Cloud and cloud shadow identification
        cloud_mask = _identify_clouds(img_arr)

        # Step 8: Write the processed image to the output directory
        output_file_path = os.path.join(dir_output_date, f'{img_dates}_{img_product}_{tile}_Fmask.tif')
        with rasterio.open(output_file_path, 'w', **profile) as dst:
            dst.write(cloud_mask, 1)


### Privite function
def _identify_clouds(img_arr):
    """
    Identify clouds and cloud-related features in the Fmask image.

    Args:
        img_arr (numpy.ndarray): Fmask image data.

    Returns:
        numpy.ndarray: Cloud mask.
    """
    cirrus = np.where(np.bitwise_and(img_arr, 0b00000001) > 0, 1, 0)
    cloud = np.where(np.bitwise_and(img_arr, 0b00000010) > 0, 1, 0)
    cloud_shadow = np.where(np.bitwise_and(img_arr, 0b00001000) > 0, 1, 0)
    adjacent_cloud = np.where(np.bitwise_and(img_arr, 0b00000100) > 0, 1, 0)
    snow_ice = np.where(np.bitwise_and(img_arr, 0b00010000) > 0, 1, 0)

    # Combine the bands representing cloud influence
    cloud_mask = np.where(cirrus + cloud + cloud_shadow + snow_ice + adjacent_cloud > 0, 1, 0)

    return cloud_mask
