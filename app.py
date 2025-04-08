import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import json

# 📂 Load labels automatically
with open("labels.json", "r") as f:
    labels = json.load(f)

# Load model
model = tf.keras.models.load_model('heavy_machine_parts_model.h5')

# Load parts database
parts_df = pd.read_csv('parts_detail.csv')

# ➡️ Create a clean column for matching
parts_df['part_name_clean'] = parts_df['part_name'].str.replace("_", " ").str.lower().str.strip()

# Function to predict
def predict_part(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Streamlit UI
st.title("🛠️ Heavy Machine Parts Classifier")

uploaded_file = st.file_uploader("Upload an image of the machine part", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = labels[predicted_class]

    # 🔥 Clean the predicted label
    predicted_label_clean = predicted_label.replace("_", " ").lower().strip()

    # Find matching part
    predicted_part = parts_df.loc[parts_df['part_name_clean'] == predicted_label_clean]

    st.subheader("🎯 Prediction Result:")

        # Show prediction probabilities
    prob_df = pd.DataFrame(predictions[0], index=labels, columns=["Probability"])
    st.dataframe(prob_df.style.highlight_max(axis=0))

    # Show predicted label
    st.success(f"**Predicted Part:** {labels[predicted_class]}")

    # ✅ Show only prediction result
    if not predicted_part.empty:
        st.success(f"✅ Part Found: {predicted_part.iloc[0]['part_name']}")
        st.write(predicted_part.drop(columns=['part_name_clean']))
    else:
        st.error("⚠️ Part not found in database.")
