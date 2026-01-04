# Cats vs Dogs Image Classification using CNN

### ✅ IBM Internship Project by **Aditya Verma**

This project uses a **Convolutional Neural Network (CNN)** built with TensorFlow to classify images as either **cats** or **dogs**. It was developed as part of the **IBM internship** and trained on the official `cats_and_dogs_filtered` dataset from TensorFlow.

## 🚀 Project Overview

- **Model Type**: CNN (Convolutional Neural Network)
- **Framework**: TensorFlow / Keras
- **Dataset**: `cats_and_dogs_filtered` (12500 training and validation images)
- **Goal**: Binary classification — Identify if the image is of a **cat** or a **dog**
- **Final Accuracy**: Achieved after 10 epochs (can be improved with tuning)

## 📂 Project Structure

cat-dog-image-classification-cnn/
── Cats_vs_Dogs_CNN_Project.ipynb ← Main Jupyter Notebook
── requirements.txt ← Required Python libraries
── dog.jpg ← Sample image for testing
── README.md ← This file
── LICENSE ← MIT License
── .gitignore ← Git ignored files

cats_and_dogs_filtered/
── train/
   ── cats/
   ── dogs/
   
── validation/
   ── cats/
   ── dogs/

🚀 How to Run

Clone this repository.
1. Install required dependencies: pip install -r requirements.txt
2. Run the notebook: Cats_vs_Dogs_CNN_Project.ipynb
3. Upload an image and test the prediction.

🧠 Model Architecture

The CNN model used in this project has the following architecture:
Input Layer (128x128x3)
↓
Conv2D (32 filters) + ReLU → MaxPooling2D
↓
Conv2D (64 filters) + ReLU → MaxPooling2D
↓
Conv2D (128 filters) + ReLU → MaxPooling2D
↓
Flatten
↓
Dense (512 units) + ReLU
↓
Output Layer: Dense (1 unit, Sigmoid activation)

Loss Function: Binary Crossentropy

Optimizer: RMSprop

Metrics: Accuracy

📊 Dataset Info

Source: TensorFlow Cats vs Dogs Dataset
Total Images:
Training: 2000 (1000 cats + 1000 dogs)
Validation: 1000 (500 cats + 500 dogs)
Image size: 128×128 pixels (resized for input)

📈 Results
Epoch	   Training Accuracy   	Validation Accuracy
 1	           ~75%	                ~70%
 10	           ~95%                 ~85%

🟢 The model learns to clearly differentiate between cat and dog images after a few epochs.
📉 Training and validation loss decrease steadily.
(These numbers are examples—replace with your real results if different.)

🐾 Sample Prediction

Uploaded Image: dog.jpg
Prediction: 🐶 Dog

💡 Future Improvements

Use Data Augmentation to reduce overfitting
Try Transfer Learning using pretrained models like MobileNet, VGG16, or ResNet
Build a web app using Streamlit or Flask for user image uploads

📄 License 

This project is licensed under the MIT License. Feel free to use, share, and modify with credit.

