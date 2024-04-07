# Binary Classification Project

This is a simple binary classification project using TensorFlow for building a neural network, Matplotlib for visualization, and OpenCV (cv2) for image processing. The goal is to classify images into two categories.

## Installation

### Requirements
- Python 3.x
- pip (Python package installer)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/binary_classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd binary_classification
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   This will install the necessary libraries including:
   - TensorFlow
   - Matplotlib
   - OpenCV (cv2)

## Dataset
- The dataset used for this project can be found in the `data` directory.
- Ensure your dataset is structured appropriately for TensorFlow's `flow_from_directory` method, where images of each class are placed in separate subdirectories.

## Usage
1. Train the model:
   ```bash
   python train_model.py
   ```
   This will train the model on the dataset and save the trained model weights.

2. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```
   This will evaluate the trained model on the validation set and display the accuracy.

3. Make predictions:
   ```bash
   python predict.py --image_path path_to_your_image.jpg
   ```
   Replace `path_to_your_image.jpg` with the path to the image you want to classify. This will use the trained model to make predictions on a single image.

## Structure
- `data/`: Directory containing the dataset.
- `train_model.py`: Script to train the neural network model.
- `evaluate_model.py`: Script to evaluate the trained model.
- `predict.py`: Script to make predictions on new images.
- `requirements.txt`: List of required Python libraries.

## Acknowledgements
- [TensorFlow](https://www.tensorflow.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://opencv.org/)
