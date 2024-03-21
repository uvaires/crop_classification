import math
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np


def calculate_sample_sizes(class_populations: list, total_samples:int, min_samples_per_class:int)-> dict:
    """
        Calculates the proportional sample sizes for each class based on their populations.

        Args:
            class_populations (list): A list of integers representing the populations of each class.
            total_samples (int): The total number of samples to distribute among the classes.
            min_samples_per_class (int): The minimum number of samples each class should have.

        Returns:
            dict: A dictionary where keys are the indices of class populations and values are the corresponding sample sizes.
        """
    total_population = sum(class_populations)

    # Calculate the proportion of each class in the total population
    class_proportions = [pop / total_population for pop in class_populations]

    # Calculate the initial samples based on a proportion
    initial_sample_sizes = [math.ceil(prop * total_samples) for prop in class_proportions]

    # Ensure each class has at least the minimum sample size
    adjusted_sample_sizes = [max(size, min_samples_per_class) for size in initial_sample_sizes]

    # Check if the adjustment resulted in a total greater than total_samples
    total_after_adjustment = sum(adjusted_sample_sizes)

    if total_after_adjustment > total_samples:
        # Identify classes with samples less than min_samples_per_class
        classes_below_min = [i for i, size in enumerate(initial_sample_sizes) if size < min_samples_per_class]

        # Replace classes with samples less than min_samples_per_class by exactly min_samples_per_class
        for i in classes_below_min:
            adjusted_sample_sizes[i] = min_samples_per_class

        # Calculate the total after replacing classes with samples less than min_samples_per_class
        total_after_min_adjustment = sum(adjusted_sample_sizes)

        # Identify the class with the most samples
        class_with_most_samples = adjusted_sample_sizes.index(max(adjusted_sample_sizes))

        # Remove excess samples from the class with the most samples to ensure a total of total_samples
        adjusted_sample_sizes[class_with_most_samples] -= total_after_min_adjustment - total_samples

    # Construct dictionary with class indices as keys and corresponding sample sizes as values
    sample_sizes_dict = {i: size for i, size in enumerate(adjusted_sample_sizes)}

    return sample_sizes_dict

def calculate_total_samples(percentage, hls_number_pixels):
    """
    Calculates the percentage values for each numerical value in the input list.

    Args:
        values (list): A list of numerical values.

    Returns:
        list: A list of percentage values corresponding to the input values.
    """
    percentage_divided = percentage/100
    total_samples = round(percentage_divided*hls_number_pixels)
    return total_samples




def count_samples_per_class(image_path):
    """
    Reads a TIFF image and calculates the total number of pixels per class.

    Args:
        image_path (str): The file path to the TIFF image.

    Returns:
        dict: A dictionary containing the total number of pixels per class, indexed by class index.
        int: The total number of pixels in the image.
    """
    # Open the TIFF image
    with rasterio.open(image_path) as src:
        # Read the image bands
        image_bands = src.read()

        # Reshape the image bands to a 2D array
        image_2d = reshape_as_image(image_bands)

    # Flatten the 2D image array
    flat_image = image_2d.reshape((-1, image_2d.shape[-1]))

    # Convert the flat image array to a list of tuples
    flat_image_tuples = [tuple(pixel) for pixel in flat_image]

    # Calculate the unique class values and their counts
    unique_values, counts = np.unique(flat_image_tuples, return_counts=True, axis=0)

    # Create a dictionary to store counts for each class
    class_counts_dict = {i: count for i, count in enumerate(counts)}

    # Calculate the total number of pixels
    total_pixels = np.sum(counts)

    return class_counts_dict, total_pixels





