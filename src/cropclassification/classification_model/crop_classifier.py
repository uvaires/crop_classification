import pandas as pd
import numpy as np
import os
import rasterio
import glob
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.impute import SimpleImputer

def training_fnn_model(base_dir:str, training_dataset:list, model_name:str, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train a Feedforward Neural Network (FNN) on given dataset and make predictions on raster layers.

    Parameters:
    - base_dir (str): Base directory path.
    - feature_data (list): List of paths to raster layers.
    - dataset (str): Path to the training dataset in Excel format.
    - model_name (str): Name of the saved FNN model file.
    - epochs (int): Number of training epochs for the FNN model.
    - batch_size (int): Batch size for training the FNN model.
    - validation_split (float): Fraction of the training data to be used as validation data.


    Returns:
    None
    """
    # Open database
    class_samples = pd.read_excel(training_dataset)
    class_samples = class_samples.dropna()

    # Select X and y
    X = class_samples.iloc[:, :-1]
    y = class_samples.iloc[:, -1]

    # Convert class labels to numerical values
    class_labels = y.astype('category')
    y = class_labels.cat.codes

    # Standardize input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build the FNN model
    fnn_model = Sequential()
    fnn_model.add(Dense(64, activation='relu', input_dim=X_scaled.shape[1]))
    fnn_model.add(Dense(32, activation='relu'))
    fnn_model.add(Dense(len(class_labels.unique()), activation='softmax'))

    # Compile the model
    fnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the learning rate
    print("Learning rate:", fnn_model.optimizer.learning_rate.numpy())

    # Train the FNN model
    history = fnn_model.fit(X_scaled, to_categorical(y), epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split)

    # Define the output directories for the model and classified image
    output_model_dir = os.path.join(base_dir, 'results', 'classified_model')
    # Save the entire FNN model
    os.makedirs(output_model_dir, exist_ok=True)
    model_file_path = os.path.join(output_model_dir, model_name)
    fnn_model.save(model_file_path)

    return model_file_path


def predicting_image(base_dir, saved_model):
    """
    Load the FNN model, make predictions, and save the classified image.

    Parameters:
    - model_file_path (str): Path to the trained FNN model file.
    - feature_layers (list): List of loaded raster layers.
    - scaler (StandardScaler): StandardScaler object used for scaling the data.
    - imputer (SimpleImputer): SimpleImputer object used for imputing missing values.
    - band_profile (dict): Profile information of the raster layers.
    - dataset (str): Path to the training dataset in Excel format.
    - image_output_dir (str): Directory to save the classified image.

    Returns:
    None
    """
    # Define the list of raster layers
    image_feature = glob.glob(os.path.join(base_dir, '**', 'inputdata', '**', '*.tif'), recursive=True)
    feature_layers, band_profile = _load_layers(image_feature)
    # Stack the feature layers using np.stack
    stacked_features = np.stack(feature_layers)

    imputer = SimpleImputer(strategy='mean')

    # Reshape the data to be 2D
    data_reshaped = stacked_features.transpose(1, 2, 0).reshape(-1, stacked_features.shape[0])
    imputer.fit(data_reshaped)
    data_reshaped = imputer.transform(data_reshaped)
    scaler = StandardScaler()
    scaler.fit(data_reshaped)
    data_reshaped = scaler.transform(data_reshaped)

    # Load the FNN model
    loaded_model = load_model(saved_model)

    # Make predictions using the loaded FNN model
    predictions = loaded_model.predict(data_reshaped)

    # Convert predictions to class labels for multi-class classification
    class_labels = np.argmax(predictions, axis=1)

    # Reshape the class labels back to the original raster shape
    class_labels_reshaped = class_labels.reshape(stacked_features.shape[1:])
    output_image_dir = os.path.join(base_dir, 'results', 'classified_images')
    # Save the classified image with the name of the Excel file
    os.makedirs(output_image_dir, exist_ok=True)
    output_path_fnn = os.path.join(output_image_dir, f'{os.path.splitext(os.path.basename(saved_model))[0]}_classified.tif')

    # Write the classified image to a GeoTIFF file
    with rasterio.open(output_path_fnn, 'w', **band_profile) as dst:
        dst.write(class_labels_reshaped, 1)



def _load_layers(feature_paths):
    """
    Load raster layers from given paths.

    Parameters:
    - feature_paths (list): List of paths to raster layers.

    Returns:
    - feature_layers (list): List of loaded raster layers.
    - band_profile (dict): Profile information of the raster layers.
    """
    feature_layers = []
    for feature_path in feature_paths:
        print(feature_path)
        with rasterio.open(feature_path) as src:
            feature_layer = src.read(1)
            band_profile = src.profile
            feature_layers.append(feature_layer)
    return feature_layers, band_profile

