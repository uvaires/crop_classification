from osgeo import gdal
import numpy as np
from scipy.ndimage import median_filter


def reclassify_raster(input_raster_path: str, output_raster_path: str, filter_size=3) -> None:
    """
        Reclassify a raster image based on a predefined reclassification dictionary and apply a median filter to reduce noise.

        Args:
            input_raster_path (str): Path to the input raster image.
            output_raster_path (str): Path to save the reclassified raster image.
        """
    # Define the reclassification dictionary with np.nan replaced by 6

    reclass_dict = {}

    for i in range(256):
        if i in [1, 2, 3, 4, 5]:
            reclass_dict[i] = i - 1
        elif 22 <= i <= 24:
            reclass_dict[i] = 5
        elif i == 41 or (47 <= i <= 63) or (76 <= i <= 127):
            reclass_dict[i] = 6
        else:
            reclass_dict[i] = 6

            # Open input raster
    input_raster = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    if input_raster is None:
        print("Error: Unable to open input raster.")
        return

    # Read a raster band
    raster_band = input_raster.GetRasterBand(1)
    if raster_band is None:
        print("Error: Unable to read raster band.")
        input_raster = None
        return

    # Convert raster to array
    raster_data = raster_band.ReadAsArray()

    # Apply median filter to the raster data
    filtered_data = median_filter(raster_data, size=filter_size)

    # Replace 'output_raster.tif' with the path to your output raster file
    output_raster = gdal.GetDriverByName('GTiff').Create(output_raster_path, input_raster.RasterXSize,
                                                         input_raster.RasterYSize, 1, gdal.GDT_Int32)
    if output_raster is None:
        print("Error: Unable to create output raster.")
        input_raster = None
        return

    # Modify the reclassified_data array based on the reclass_dict
    reclassified_data = np.vectorize(reclass_dict.get, otypes=[int])(filtered_data)
    output_raster.GetRasterBand(1).WriteArray(reclassified_data)
    output_raster.SetProjection(input_raster.GetProjection())
    output_raster.SetGeoTransform(input_raster.GetGeoTransform())
    output_raster.FlushCache()

    # Close input and output rasters
    input_raster = None
    output_raster = None
