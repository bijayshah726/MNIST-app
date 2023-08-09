# MNIST Digit Classifier with Grad-CAM

This project demonstrates a simple MNIST digit classifier using a deep learning model, along with Grad-CAM visualization to highlight important regions in the input image. The project also includes a Streamlit web application for interactive use.

## Table of Contents
- [Overview](#overview)
- [Installation and Usage](#installation)
- [Streamlit Deployment](#streamlit-deployment)

## Overview

The project consists of the following components:

1. A simple image classifier using TensorFlow/Keras with convolutional layers.
2. Grad-CAM implementation to visualize important regions in the input image.
3. A Streamlit web application to upload images, view predictions, and visualize Grad-CAM heatmaps.

Streamlit UI:
![Streamlit UI](https://github.com/bijayshah726/MNIST-app/assets/89373352/5dbf65f3-671f-4df4-acfd-b46c7d3f5415)

Overlayed HeatMap with digit 3:
![image](https://github.com/bijayshah726/MNIST-app/assets/89373352/7d40c4c9-bd0c-46dc-acce-f5a70e455b91)


GradCam HeatMap:
![image](https://github.com/bijayshah726/MNIST-app/assets/89373352/6cda8ea4-9aec-47fe-8d4a-2da371d45ad9)




## Installation and Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/bijayshah726/mnist-app.git

   cd mnist-app   #Navigate to local directory
   python -m venv venv   #Optional (but recommended to create virtual environment)
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   streamlit run app.py  #opens the default browser where the streamlit app can be viewed
   #Upload an image using the web app and click "Predict" to see the predicted class and Grad-CAM heatmap overlay.
   ```

2. Use the following URL to view the streamlit app
   https://mnist-gradcam.streamlit.app/   


## Streamlit Deployment
Sign up or log in to Streamlit Sharing.

Fork or clone this GitHub repository to your GitHub account.

Connect your GitHub repository to Streamlit Sharing.

Deploy the app on Streamlit Sharing (more details [here](https://docs.streamlit.io/streamlit-community-cloud/share-your-app))



Contributions and suggestions are welcome! Feel free to open issues or submit pull requests if you encounter any problems or have improvements to suggest.
   
