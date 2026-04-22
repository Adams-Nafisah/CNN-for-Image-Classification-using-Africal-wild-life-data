# CNN-for-Image-Classification-using-Africal-wild-life-data

# African Wildlife Image Classification Project

## Project Overview

This project aims to develop a Convolutional Neural Network (CNN) model for classifying images of African wildlife into four categories: 'buffalo', 'elephant', 'rhino', and 'zebra'. The project explores various stages of model development, from initial data loading and preprocessing to model architecture design, training, evaluation, and strategies to mitigate overfitting.

## Dataset

The dataset consists of images of African wildlife, organized into `train`, `val`, and `test` splits. The class labels are embedded within the image filenames and defined in an `african-wildlife.yaml` file.

### Data Loading and Preprocessing

1.  **Mount Google Drive**: The dataset, stored as a `.zip` file, is loaded from Google Drive.
2.  **Extract Dataset**: The `.zip` file is extracted to `/content/dataset`.
3.  **Directory Structure Check**: Initial inspection revealed a complex directory structure that standard TensorFlow utilities (e.g., `image_dataset_from_directory`) could misinterpret.
4.  **Class Name Extraction**: The `african-wildlife.yaml` file was parsed to correctly identify the four class names and their corresponding integer labels.
5.  **Custom Data Pipeline**: A custom `tf.data.Dataset` pipeline was implemented to correctly load images and their 0-indexed labels, addressing initial issues where the model achieved 100% accuracy due to misinterpreting directory names as classes.
6.  **Image Preprocessing**: Images are resized to 180x180 pixels.

## Model Architecture

The model is a sequential CNN built with `tf.keras` layers:

*   **Input**: Images of size (180, 180, 3).
*   **Data Augmentation**: `RandomFlip`, `RandomRotation`, and `RandomZoom` layers are applied to increase data diversity and reduce overfitting.
*   **Rescaling**: Pixels are rescaled to the [0, 1] range.
*   **Convolutional Layers**: Three `Conv2D` layers with decreasing filter sizes (8, 16, 32) and `MaxPooling2D` are used to extract features.
*   **Dropout**: A `Dropout(0.3)` layer is included for regularization.
*   **Flatten**: Converts the 2D feature maps to a 1D vector.
*   **Dense Layers**: A `Dense` layer with 64 neurons and a final `Dense` layer with `num_classes` (4) output neurons.

## Training and Evaluation

*   **Optimizer**: Adam optimizer.
*   **Loss Function**: `SparseCategoricalCrossentropy(from_logits=True)`.
*   **Metrics**: Accuracy.
*   **Early Stopping**: An `EarlyStopping` callback (`monitor='val_loss'`, `patience=5`, `restore_best_weights=True`) was implemented to prevent overfitting by stopping training when validation loss stops improving.

## Challenges and Solutions

*   **Problem: Incorrect Data Interpretation**: Initially, the model achieved 100% accuracy due to `tf.keras.utils.image_dataset_from_directory` misinterpreting the dataset structure. 
    *   **Solution**: Implemented a custom data loading pipeline using `tf.data.Dataset` to correctly parse image paths and extract 0-indexed labels from filenames and the YAML configuration.
*   **Problem: Overfitting**: After correcting data loading, the model showed signs of overfitting (high training accuracy/low training loss, but stagnant/increasing validation loss).
    *   **Solution 1**: Introduced data augmentation (random flips, rotations, zooms).
    *   **Solution 2**: Added `Dropout` layers.
    *   **Solution 3**: Reduced model complexity by decreasing the number of filters in `Conv2D` layers and neurons in `Dense` layers.
    *   **Solution 4**: Increased the `Dropout` rate from 0.2 to 0.3.
    *   **Solution 5**: Implemented `EarlyStopping` to stop training when validation loss worsens, effectively preventing the model from memorizing training data.

## Results

After applying all mitigation strategies, the model typically trained for around 6 epochs before early stopping. The final validation accuracy hovered between **45-55%**, with validation loss showing signs of instability. This indicates that while overfitting was addressed, the current CNN architecture might not be powerful enough to achieve higher generalization performance on this dataset.

## Reproducibility

To reproduce the results, ensure you have:

1.  **Google Colab Environment**: The notebook is designed to run in Google Colab.
2.  **Dataset**: Upload the `african-wildlife.zip` dataset to your Google Drive and ensure it's accessible at `/content/drive/MyDrive/african-wildlife.zip`.
3.  **Run Cells Sequentially**: Execute all cells in the notebook in order. The notebook includes all necessary imports, data loading, preprocessing, model definition, training, and evaluation steps.

## Future Work

Given the current performance, **Transfer Learning** using a pre-trained model (e.g., MobileNet, ResNet) on a large dataset like ImageNet, and then fine-tuning it on this specific African wildlife dataset, is highly recommended to achieve significantly better accuracy and generalization.
