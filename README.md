# NullClass Internship Assignment

## Drowsiness Detection System

### Overview

This repository contains the code and resources for a real-time drowsiness detection system using facial landmarks and a deep learning model. The system uses a pre-trained model to classify whether a person's eyes are open or closed and triggers an alarm if drowsiness is detected.

### Files

- `alarm.wav`: Audio file for the alarm sound.
- `drowsiness.py`: Python script for real-time drowsiness detection.
- `model-creation.ipynb`: Jupyter Notebook containing the code for creating the drowsiness detection model.
- `models.h5`: Pre-trained deep learning model for eye state classification.
- `practical_vdo.mp4`: Sample video demonstrating the drowsiness detection system.
- `shape_predictor_68_face_landmarks.dat`: Pre-trained facial landmarks predictor for detecting eyes.

### Dataset

The dataset used for training the model is not included in this repository due to size limitations. You can access the dataset [here](https://www.kaggle.com/datasets/hazemfahmy/openned-closed-eyes).

### Instructions

Ensure you have the following dependencies installed:

- `opencv-python`
- `numpy`
- `dlib`
- `scipy`
- `pygame`
- `keras`

#### You can install them using:
pip install opencv-python numpy dlib scipy pygame keras

#### Run the drowsiness.py script for real-time drowsiness detection.
python drowsiness.py

### Demo
Watch the [demo video](https://drive.google.com/file/d/1R6gGEy7FRPeCHvSs0fcTw2JvN_WvjNxK/view?usp=sharing) to see the drowsiness detection system in action.

### Credits
This project was developed as part of the NullClass Internship.



