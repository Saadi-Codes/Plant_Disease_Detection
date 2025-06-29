import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# Loaded trained model
MODEL_URL = "https://huggingface.co/Saadi-Codes-22/plant-disease-model/resolve/main/trained_model.keras"
MODEL_PATH = "trained_model.keras"

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

model = tf.keras.models.load_model(MODEL_PATH)

# Defined class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def predict(img):
    try:
        print("📸 Image received.")
        image = img.resize((128, 128))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        print("🔍 Image preprocessed.")

        predictions = model.predict(image)
        print(f"📊 Predictions: {predictions}")

        index = np.argmax(predictions)
        print(f"✅ Predicted class index: {index}")
        return f"🌿 Prediction: Image consists of a {class_names[index]} leaf"

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Something went wrong. Please try again."

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="Plant Disease Detection",
    description="Upload an image of a plant leaf to identify possible disease."
)

if __name__ == "__main__":
    interface.launch()
