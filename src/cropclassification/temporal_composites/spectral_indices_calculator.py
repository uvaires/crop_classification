import os
import glob
import numpy as np
import rasterio

def spectral_indices(base_dir)->None:
    """
    Processes HLS images, calculates spectral indices, and exports them to specified directories.

    Args:
        input_dir (str): Root directory containing HLS percentiles
        output_dir (str): Root directory to export the spectral indices
        percentiles_folder (str): Folder containing the HLS percentiles

    Returns:
        None
    """
    hls_imgs = glob.glob(os.path.join(base_dir, '**', 'percentiles', '**', '*B02_p*.tif'), recursive=True)
    for hls_img in hls_imgs:
        print(hls_img)
        img_index = os.path.basename(hls_img).split('_')[0]
        basedir_indices = os.path.join(base_dir, 'temporal_composites', 'inputdata', 'spectral_indexes')
        dir_output_date = os.path.join(basedir_indices, img_index)
        os.makedirs(dir_output_date, exist_ok=True)

        blue_band_hls = hls_img
        red_band_hls = hls_img.replace('B02', 'B04')
        nir_band_hls = hls_img.replace('B02', 'NIR')
        swr1_band_hls = hls_img.replace('B02', 'SWR1')
        swr2_band_hls = hls_img.replace('B02', 'SWR2')

        blue_band, _ = _read_raster(blue_band_hls)
        del blue_band_hls
        red_band, _ = _read_raster(red_band_hls)
        del red_band_hls
        nir_band, src_profile = _read_raster(nir_band_hls)
        del nir_band_hls
        swr1_band, _ = _read_raster(swr1_band_hls)
        del swr1_band_hls
        swr2_band, _ = _read_raster(swr2_band_hls)
        del swr2_band_hls

        evi = _calculate_evi(nir_band, red_band, blue_band)
        del blue_band, red_band
        ndbi = _calculate_ndbi(swr1_band, nir_band)
        ndwi = _calculate_ndwi(nir_band, swr1_band)
        nbr = _calculate_nbr(nir_band, swr2_band)
        del nir_band
        ndti = _calculate_ndti(swr1_band, swr2_band)
        del swr1_band, swr2_band

        evi_output = os.path.join(dir_output_date, os.path.basename(hls_img).replace('B02', 'evi'))
        ndbi_output = os.path.join(dir_output_date, os.path.basename(hls_img).replace('B02', 'ndbi'))
        ndwi_output = os.path.join(dir_output_date, os.path.basename(hls_img).replace('B02', 'ndwi'))
        nbr_output = os.path.join(dir_output_date, os.path.basename(hls_img).replace('B02', 'nbr'))
        ndti_output = os.path.join(dir_output_date, os.path.basename(hls_img).replace('B02', 'ndti'))

        _export_index_to_drive(evi, evi_output, src_profile)
        _export_index_to_drive(ndbi, ndbi_output, src_profile)
        _export_index_to_drive(ndwi, ndwi_output, src_profile)
        _export_index_to_drive(nbr, nbr_output, src_profile)
        _export_index_to_drive(ndti, ndti_output, src_profile)



### Private functions #####

def _read_raster(input_raster):
    """
    Reads raster image using rasterio and returns rasterband and profile.

    Args:
        input_raster (str): Path to the input raster image.

    Returns:
        tuple: A tuple containing rasterband and source profile.
    """
    with rasterio.open(input_raster) as src:
        rasterband = src.read(1)
        src_profile = src.profile
    return rasterband, src_profile

def _calculate_evi(nir_band, red_band, blue_band):
    """
    Calculates the Enhanced Vegetation Index (EVI) using the specified bands.

    Args:
        nir_band (numpy.ndarray): Near-Infrared band.
        red_band (numpy.ndarray): Red band.
        blue_band (numpy.ndarray): Blue band.

    Returns:
        numpy.ndarray: Computed EVI values.
    """
    evi_numerator = nir_band - red_band
    evi_denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
    mask = evi_denominator != 0
    evi = np.zeros_like(nir_band, dtype=np.float32)
    evi[mask] = 2.5 * evi_numerator[mask] / evi_denominator[mask]
    return evi

def _calculate_ndbi(swr1_band, nir_band):
    """
    Calculates the Normalized Difference Built-Up Index (NDBI) using specified bands.

    Args:
        swr1_band (numpy.ndarray): Shortwave Infrared band 1.
        nir_band (numpy.ndarray): Near-Infrared band.

    Returns:
        numpy.ndarray: Computed NDBI values.
    """
    ndbi_numerator = swr1_band - nir_band
    ndbi_denominator = swr1_band + nir_band
    mask = ndbi_denominator != 0
    ndbi = np.zeros_like(swr1_band, dtype=np.float32)
    ndbi[mask] = ndbi_numerator[mask] / ndbi_denominator[mask]
    return ndbi

def _calculate_ndwi(nir_band, swr1_band):
    """
    Calculates the Normalized Difference Water Index (NDWI) using specified bands.

    Args:
        nir_band (numpy.ndarray): Near-Infrared band.
        swr1_band (numpy.ndarray): Shortwave Infrared band 1.

    Returns:
        numpy.ndarray: Computed NDWI values.
    """
    ndwi_numerator = nir_band - swr1_band
    ndwi_denominator = nir_band + swr1_band
    mask = ndwi_denominator != 0
    ndwi = np.zeros_like(nir_band, dtype=np.float32)
    ndwi[mask] = ndwi_numerator[mask] / ndwi_denominator[mask]
    return ndwi

def _calculate_nbr(nir_band, swr2_band):
    """
    Calculates the Normalized Burn Ratio (NBR) using specified bands.

    Args:
        nir_band (numpy.ndarray): Near-Infrared band.
        swr2_band (numpy.ndarray): Shortwave Infrared band 2.

    Returns:
        numpy.ndarray: Computed NBR values.
    """
    nbr_numerator = nir_band - swr2_band
    nbr_denominator = nir_band + swr2_band
    mask = nbr_denominator != 0
    nbr = np.zeros_like(nir_band, dtype=np.float32)
    nbr[mask] = nbr_numerator[mask] / nbr_denominator[mask]
    return nbr

def _calculate_ndti(swr1_band, swr2_band):
    """
    Calculates the Normalized Difference Turbidity Index (NDTI) using specified bands.

    Args:
        swr1_band (numpy.ndarray): Shortwave Infrared band 1.
        swr2_band (numpy.ndarray): Shortwave Infrared band 2.

    Returns:
        numpy.ndarray: Computed NDTI values.
    """
    ndti_numerator = swr1_band - swr2_band
    ndti_denominator = swr1_band + swr2_band
    mask = ndti_denominator != 0
    ndti = np.zeros_like(swr1_band, dtype=np.float32)
    ndti[mask] = ndti_numerator[mask] / ndti_denominator[mask]
    return ndti

def _export_index_to_drive(index, output_path, src_profile):
    """
    Exports the calculated index to a GeoTIFF file.

    Args:
        index (numpy.ndarray): Index values to be exported.
        output_path (str): Output path for the GeoTIFF file.
        src_profile (dict): Source profile of the raster.

    Returns:
        None
    """
    src_profile['dtype'] = 'float32'
    src_profile['compress'] = 'lzw'
    src_profile['nodata'] = np.nan
    with rasterio.open(output_path, 'w', **src_profile) as dst:
        dst.write(index, 1)

