from typing import Dict
import os
import glob
from datetime import datetime, timedelta
import rasterio


def organize_hls(hls_raw_images: str, base_dir: str, tile_dir: str, start_date: str, end_date: str) -> None:
    """
    Rename HLS bands and organize them in a specified directory structure.

    Args:
        hls_raw_images (str): Folder to get images from.
        base_dir (str): Main output directory for organized HLS bands.
        tile_dir (str): Directory containing tile-specific subdirectories.
        start_date (str): Start date of the range in the format 'YYYYMMDD'.
        end_date (str): End date of the range in the format 'YYYYMMDD'.
    """
    # Select only the interested bands
    bands_per_product = {
        'S30': {'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'Fmask'},
        'L30': {'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask'}
    }
    # Rename the bands based on their number
    band_names = {'S30': {'B08': 'NIR', 'B11': 'SWR1', 'B12': 'SWR2'},
                  'L30': {'B05': 'NIR', 'B06': 'SWR1', 'B07': 'SWR2'}}

    # Convert start_date and end_date strings to datetime objects
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    # Globbing Input Files
    hls_images = glob.glob(os.path.join(hls_raw_images, tile_dir, '**', '*.tif'), recursive=True)
    hls_tile_image = {file_path: os.path.basename(file_path).split('.')[2] for file_path in hls_images}
    target_crs = _get_valid_crs(hls_images)

    # Within the loop where each image file is processed
    for file_path, tile in hls_tile_image.items():

        # Extracting Metadata
        metadata = _extract_metadata_from_path(file_path)

        # Convert metadata date string to datetime
        metadata_date = datetime.strptime(metadata['formatted_date'], '%Y%m%d')

        # Check if the date is within the specified range
        if start_date <= metadata_date <= end_date:
            # Extract band name from the file name
            band_name_hls = os.path.basename(file_path).split('.')[6]

            if band_name_hls in bands_per_product.get(metadata['hls_product'], {}):
                # Check if the band name is present in the bands_per_product dictionary for the current HLS product
                if band_name_hls in band_names.get(metadata['hls_product'], {}):
                    # If band name has a specific name defined in band_names, use it for renaming
                    band_name = band_names[metadata['hls_product']][band_name_hls]
                else:
                    # Otherwise, use the original band name
                    band_name = band_name_hls

                # Create a directory to store the data
                dir_output = os.path.join(base_dir, 'pre_process', 'hls_organized')
                # Continue processing only if the date is within the range
                hls_renamed = f"{metadata['formatted_date']}_{metadata['hls_product']}_{tile}_{band_name}.tif"
                dir_output_date = os.path.join(dir_output, metadata['date_string'])
                os.makedirs(dir_output_date, exist_ok=True)

                with rasterio.open(file_path) as src:
                    band_data = src.read(1)
                    src_profile = src.profile

                src_profile['crs'] = target_crs

                if band_name == 'Fmask':
                    src_profile['dtype'] = 'uint8'
                else:
                    src_profile['dtype'] = 'uint16'

                hls_output_path = os.path.join(dir_output_date, hls_renamed)
                src_profile['nodata'] = 0
                src_profile['compress'] = 'lzw'

                with rasterio.open(hls_output_path, 'w', **src_profile) as dst:
                    dst.write(band_data, 1)


### Privite Functions ###
def _is_valid_utm_crs(crs):
    """
    Check if the provided CRS is a valid UTM projection.

    Args:
        crs (rasterio.crs.CRS): Coordinate Reference System.

    Returns:
        bool: True if the CRS is a valid UTM projection, False otherwise.
    """
    try:
        epsg_code = crs.to_epsg()
        return 32601 <= epsg_code <= 32760
    except:
        return False


def _extract_metadata_from_path(file_path: str) -> Dict[str, str]:
    """
    Find and return the valid UTM CRS among a list of HLS image paths.

    Args:
        hls_images_paths (List[str]): List of paths to HLS images.

    Returns:
        str: Valid UTM CRS in the format 'EPSG:xxxx' if found, otherwise an empty string.
    """
    julian_date = os.path.basename(file_path).split('.')[3].split('T')[0]
    year = int(julian_date[:4])
    day_of_year = int(julian_date[4:])
    image_date = datetime(year, 1, 1) + timedelta(day_of_year - 1)
    date_string = image_date.strftime('%Y%m%d')
    formatted_date = image_date.strftime('%Y%m%d')
    hls_product = os.path.basename(file_path).split('.')[1]
    metadata = {
        'date_string': date_string,
        'formatted_date': formatted_date,
        'hls_product': hls_product,
    }
    return metadata


def _get_valid_crs(hls_images_paths) -> str:
    """
        Find and return the valid UTM CRS among a list of HLS image paths.

        Args:
            hls_images_paths (List[str]): List of paths to HLS images.

        Returns:
            str: Valid UTM CRS in the format 'EPSG:xxxx' if found, otherwise an empty string.
        """
    for path in hls_images_paths:
        with rasterio.open(path) as src:
            crs = src.crs
        if _is_valid_utm_crs(crs):
            return crs



