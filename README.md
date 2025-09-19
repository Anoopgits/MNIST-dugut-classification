# MNIST Digit Classification Project

## Project Overview
This project demonstrates a **handwritten digit recognition system** using the **MNIST dataset**.  
It uses **Convolutional Neural Networks (CNN)** implemented in **TensorFlow/Keras** to classify digits from 0 to 9.  
A **Streamlit web app** is also included to allow users to upload images and get predictions in real-time.

---

## Features
- Train a CNN model on MNIST dataset
- Achieve high accuracy on test data
- Web interface using **Streamlit**
- Users can input images and get predicted digit
- Simple, clean, and interactive UI

---

## Dataset
- **MNIST Handwritten Digits**
- 60,000 training images, 10,000 test images
- Images are grayscale, 28x28 pixels
- Dataset is available via TensorFlow/Keras API:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
Installation

Clone the repository:

git clone <repository_url>
cd mnist-digit-classification


Create a virtual environment:

python -m venv myenv
myenv\Scripts\activate


Install required libraries:

pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python streamlit

How to Run
1️⃣ Train Model
python train_model.py

2️⃣ Run Streamlit App
streamlit run app.py


Open the URL shown in the terminal to interact with the web app

Upload an image of a digit and see the predicted result

File Structure
mnist-digit-classification/
│
├── app.py             # Streamlit app
├── train_model.py     # Script to train CNN model
├── mnist_model.h5     # Trained model (optional)
├── requirements.txt   # Required Python libraries
├── README.md          # Project documentation
└── images/            # Sample input images

Model Details

Architecture: CNN

Layers: Conv2D → MaxPooling → Flatten → Dense

Activation: ReLU (hidden layers), Softmax (output)

Optimizer: Adam

Loss: SparseCategoricalCrossentropy

Metrics: Accuracy

License

This project is open-source and free to use.

Author=>
Anoop Ojha

