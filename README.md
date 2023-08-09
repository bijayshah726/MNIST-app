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



## Installation and Usage

### Web App
   You can directly view the app and use it for prediction and visualization.
   https://mnist-gradcam.streamlit.app/ 

### Local Machine
   #### Prerequisites

- An Operating System like Windows, OsX or Linux
- A working [Python](https://www.python.org/) installation.
- a Shell
  - [Git Bash](https://git-scm.com/downloads) for Windows 8.1
  - [wsl](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) for For Windows 10
- an Editor
  - [VS Code](https://code.visualstudio.com/) (Preferred) or [PyCharm](https://www.jetbrains.com/pycharm/).
- [Git cli](https://git-scm.com/downloads)

#### Installation

Clone the repo

```bash
git clone https://github.com/bijayshah726/MNIST-app.git
```

cd into the project root folder

```bash
cd MNIST-app
```

##### Create virtual environment

###### via python

Then you should create a virtual environment named .venv

```bash
python -m venv .venv
```

and activate the environment.

On Linux, OsX or in a Windows Git Bash terminal it's

```bash
source .venv/Scripts/activate
```

or alternatively

```bash
source .venv/bin/activate
```

In a Windows terminal it's

```bash
.venv/Scripts/activate.bat
```

###### or via anaconda

Create virtual environment named MNIST-app

```bash
conda create -n MNIST-app python
```

and activate environment.

```bash
activate MNIST-app
```

If you are on windows you need to install some things required by GeoPandas by following [these instructions](https://geoffboeing.com/2014/09/using-geopandas-windows/).

Then you should install the local requirements

```bash
pip install -r requirements.txt
```

#### Build and run the Application Locally

```bash
streamlit run app.py  #opens the default browser where the streamlit app can be viewed
#Upload an image using the web app and click "Predict" to see the predicted class and Grad-CAM heatmap overlay.
```


  


## Streamlit Deployment
Sign up or log in to Streamlit Sharing.

Fork or clone this GitHub repository to your GitHub account.

Connect your GitHub repository to Streamlit Sharing.

Deploy the app on Streamlit Sharing (more details [here](https://docs.streamlit.io/streamlit-community-cloud/share-your-app))



Contributions and suggestions are welcome! Feel free to open issues or submit pull requests if you encounter any problems or have improvements to suggest.
   
