---
title: Plant Disease Detector 🌿
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.21.0"
app_file: app.py
pinned: false
---

## 🌿 Plant Disease Detection Web App

This is a Streamlit-based web application that uses a trained Convolutional Neural Network (CNN) to detect plant diseases from leaf images. The model is built with TensorFlow/Keras and trained on the **New Plant Diseases Dataset (Augmented)** from Kaggle.

---

## Features

- Upload a leaf image (JPG, PNG) from given test images
- Predict plant disease using a deep learning model
- Shows top prediction with class name
- Clean UI with real-time results

---

## File Structure

Plant Disease Detection/

├── images
    ├── main image
├── .streamlit
    ├── config.toml
├── test
    ├──(29 seperate image files)
├── Plant_Disease_detection.ipynb
├── Plant_disease_detection.png
├── main.py
├── trained_model.keras
├── training_history.json
├── requirements.txt
├── README.md

---

## Dataset Info

- Source: Kaggle - New Plant Diseases Dataset (Augmented)
- Number of Classes: 38
- Input Image Size: 128x128

## Credits

- Developed by **Saad Ahmed**
- Model trained in Google Colab
- UI built using Streamlit
