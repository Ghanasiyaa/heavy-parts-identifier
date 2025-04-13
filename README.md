# heavy-parts-Identifier

A deep learning web app that classifies images of heavy machine parts using a pre-trained TensorFlow model, built and deployed with Streamlit.

## Demo
[Live Demo Link](https://your-streamlit-app-url)

## Project Overview

# Heavy Machine Parts Classifier

A deep learning web app that classifies images of heavy machine parts using a pre-trained TensorFlow model, built and deployed with Streamlit.

Dependencies

pip install -r requirements.txt

Run the Streamlit app locally:

streamlit run app.py

## Flow Diagram

User (Upload Image)
        ↓
Streamlit Web App (app.py)
        ↓
Image Preprocessing (Resize, Normalize, Expand dims)
        ↓
Pre-trained Model (.h5 file, TensorFlow)
        ↓
Prediction
        ↓
Result Displayed (Predicted Class)
