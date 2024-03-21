import geopandas as gpd
import rasterio
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)



def evaluation_metrics(base_dir, samples_points:str, predicted_img:str, metrics_output:str, confusin_matrix_output:str,
                       class_labels: list, plot_title:str)->None:
    """
    Main function to perform the classification evaluation and visualization.

    Parameters:
    - samples_points (str): Path to the point samples shapefile.
    - predicted_img (str): Path to the classified image.
    - path_save_metrics_excel (str): Path to save the metrics Excel file.
    - path_confusin_matrix (str): Path to save the confusion matrix plot.
    - class_labels (list): List of class labels.
    - plot_title (str): Title of the confusion matrix plot.
    """
    # Define the directory to storage evaluation metrics
    metrics = os.path.join(base_dir, 'results', 'model_evaluation', 'metrics')
    # Create the output directory if it doesn't exist
    os.makedirs(metrics, exist_ok=True)
    # Export the metrics as Excel file
    path_save_metrics_excel = os.path.join(metrics, metrics_output)

    # Define the directory to storage the confusion matrix
    confusion_matrix_path = os.path.join(base_dir, 'results', 'model_evaluation', 'confusion_matrix')
    # Create the output directory if it doesn't exist
    os.makedirs(confusion_matrix_path, exist_ok=True)
    # Save the confusion matrix
    path_confusin_matrix = os.path.join(confusion_matrix_path, confusin_matrix_output)

    # Read points
    gdf = gpd.read_file(samples_points)

    class_counts = gdf['value'].value_counts()

    # Display the counts for each class
    print("Class Counts:")
    print(class_counts)

    # List of raster paths
    raster_paths = [predicted_img]

    # Apply function to extract the raster values to point for the predicted image
    result_df = _extract_raster_values_to_dataframe(raster_paths, gdf)

    # Convert the GeoDataFrame to a regular DataFrame
    result_df = result_df.drop(columns='geometry')

    # Apply a lambda function to convert values to regular numbers
    result_df = result_df.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    # Assuming the first column is observed data and the second column is predicted values
    observed_data = result_df.iloc[:, 0]
    predicted_data = result_df.iloc[:, 1]

    # Convert values to integers (assuming they are class labels)
    predicted_data = predicted_data.astype(int)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(observed_data, predicted_data)


    # Call plot_confusion_matrix with the provided class_labels and plot_title
    _plot_confusion_matrix(conf_matrix, class_labels, title=plot_title,
                          save_path=path_confusin_matrix, dpi=300)

    # Save metrics
    _save_metrics_to_excel(observed_data, predicted_data, path_save_metrics_excel)



## Privite functions
def _extract_raster_values_to_dataframe(img_paths, gdf):
    """
    Extract raster values for each point in a GeoDataFrame and create a new DataFrame.

    Parameters:
    - img_paths (list): List of paths to raster images.
    - gdf (GeoDataFrame): GeoDataFrame containing point geometries.

    Returns:
    - result_df (DataFrame): DataFrame with extracted raster values.
    """
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
                                     columns=[f"Raster_{os.path.basename(raster_path)}_{i}" for i in
                                              range(len(values_list[0]))])

            # Concatenate the new DataFrame to the result_df
            result_df = pd.concat([result_df, raster_df], axis=1)

    return result_df


def _save_metrics_to_excel(y_true, y_pred_class, metrics_excel_path):
    """
    Calculate classification metrics and save them to an Excel file.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred_class (array-like): Predicted labels.
    - metrics_excel_path (str): Path to save the metrics Excel file.
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, average='weighted')
    recall = recall_score(y_true, y_pred_class, average='weighted')
    f1 = f1_score(y_true, y_pred_class, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred_class)
    class_report = classification_report(y_true, y_pred_class, output_dict=True)

    # Calculate producer accuracy and user accuracy
    producer_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    user_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)

    # Calculate average producer accuracy and user accuracy
    avg_producer_accuracy = np.mean(producer_accuracy)
    avg_user_accuracy = np.mean(user_accuracy)

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'Average Producer Accuracy': [avg_producer_accuracy],
        'Average User Accuracy': [avg_user_accuracy],
    })

    # Add confusion matrix to the DataFrame
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[f'Predicted_{i}' for i in range(conf_matrix.shape[1])])
    conf_matrix_df.index = [f'Actual_{i}' for i in range(conf_matrix.shape[0])]
    metrics_df = pd.concat([metrics_df, conf_matrix_df], axis=1)

    # Add class-specific metrics to the DataFrame
    class_metrics_df = pd.DataFrame.from_dict({int(k): v for k, v in class_report.items() if k.isdigit()})
    metrics_df = pd.concat([metrics_df, class_metrics_df.transpose()], axis=1)

    # Save metrics to an Excel file
    metrics_df.to_excel(metrics_excel_path, index=False)


def _plot_confusion_matrix(conf_matrix, class_labels, title='Confusion Matrix', highlight_diagonal=True, save_path=None,
                          dpi=300, width=6.5, height=5):
    """
    Plot a confusion matrix.

    Parameters:
    - conf_matrix (array-like): Confusion matrix.
    - class_labels (list): List of class labels.
    - title (str): Title of the plot.
    - highlight_diagonal (bool): Whether to highlight the diagonal cells.
    - save_path (str): Path to save the plot.
    - dpi (int): Dots per inch for the plot resolution.
    - width (float): Width of the plot.
    - height (float): Height of the plot.
    """
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Create a figure and axis with specified width and height
    fig, ax = plt.subplots(figsize=(width, height))

    # Set color for the diagonal cells
    cmap = sns.light_palette("navy", as_cmap=True) if highlight_diagonal else 'Blues'

    # Plot the confusion matrix with adjustments
    sns.heatmap(conf_matrix, annot=True, cmap=cmap, fmt="d", cbar=True,  # Set cbar to False
                xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=1.5, square=False)

    # Set labels, title, and ticks
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left')

    # Set tick positions to the center of the cells
    ax.set_xticks(np.arange(len(class_labels)) + 0.5)
    ax.set_yticks(np.arange(len(class_labels)) + 0.5)

    # Set background color for better contrast
    ax.set_facecolor('lightgray')

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()


