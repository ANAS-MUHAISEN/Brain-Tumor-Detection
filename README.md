Brain Tumor Detection using YOLOv11
Introduction
This project is aimed at detecting brain tumors using a deep learning model based on YOLOv11 (You Only Look Once), a state-of-the-art object detection algorithm. The model is trained using a Medical Image Dataset consisting of MRI images of brain tumors. The dataset is categorized into four classes:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The model's goal is to identify and classify the tumor type or determine if the image contains no tumor.

Table of Contents
Installation

Dataset

Model Training

Evaluation

Results

Usage

Acknowledgements

Installation
To get started with the project, follow the instructions below:

Prerequisites
Make sure you have the following installed:

Python 3.7 or higher

PyTorch 2.6.0 or higher

Ultralytics YOLO (for the YOLO model)

OpenCV (for image processing)

Matplotlib (for plotting)

You can install the dependencies using pip:

bash
Copy
Edit
pip install torch torchvision matplotlib opencv-python ultralytics
Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
Dataset
The dataset used for this project is sourced from Roboflow and contains a total of 3,903 MRI images of brain tumors. The images are categorized into the following classes:

Glioma

Meningioma

Pituitary Tumor

No Tumor

The dataset is divided into the following sets:

Training Set (70%)

Validation Set (20%)

Test Set (10%)

You can download the dataset from the Kaggle Brain Tumor Detection dataset or use a custom dataset that follows the same structure.

Model Training
To train the YOLOv11 model, follow these steps:

Prepare the dataset:

Ensure the dataset is split into the appropriate directories for training, validation, and testing.

Each image should have a corresponding annotation file in the YOLO format (bounding boxes with class labels).

Train the model:

Use the following command to start training:

bash
Copy
Edit
from ultralytics import YOLO

# Load YOLOv8 model (v11 is based on the same YOLOv8 family)
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='data.yaml', epochs=30, imgsz=640)
data.yaml should be a configuration file pointing to the dataset's directories for train, validation, and test images.

Model configuration:

Make sure you specify the correct image size (imgsz) and number of epochs for optimal training performance.

Evaluation
After training the model, you can evaluate its performance on the test dataset:

python
Copy
Edit
results = model.evaluate('path_to_test_images')
print(f"Accuracy: {results.metrics['mAP50']}")
The results will show the accuracy of the model, including metrics such as Mean Average Precision (mAP).

Results
The model will output predictions in the form of bounding boxes, class labels, and confidence scores. You can visualize these predictions using the following code:

python
Copy
Edit
results = model.predict("path_to_image.jpg", save=True)
results[0].plot()
This will display the image with the detected tumor and its classification.

Usage
Test Image Prediction
You can use the trained model to predict whether a new MRI image contains a tumor or not, as shown below:

python
Copy
Edit
results = model.predict("path_to_new_image.jpg", save=True)
results[0].plot()  # Show the image with predicted bounding boxes and labels
Save the Model
You can save the trained model using the following code:

python
Copy
Edit
model.save('model.pt')
To load the model later, use:

python
Copy
Edit
model = YOLO('model.pt')
Acknowledgements
The dataset is from Roboflow Universe and Kaggle.

The YOLOv11 model is implemented by Ultralytics.

Special thanks to the developers of the libraries used in this project: PyTorch, OpenCV, Matplotlib.

License
This project is licensed under the MIT License - see the LICENSE file for details.

