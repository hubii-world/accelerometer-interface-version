# Emotion Recognition Pipeline Using Movement Data

This pipeline enables users to **train machine learning models** on their own movement data and **classify movements** using a trained model. Designed for **researchers and non-experts**, it simplifies preprocessing, feature extraction, and model training, allowing for quick and efficient emotion recognition based on smartwatch accelerometer data.

## 🚀 Features
- ✅ **Train models** on labeled movement data  
- ✅ **Classify new movement data** using a pre-trained model  
- ✅ **Automatic feature extraction** (time, frequency, statistical, and spatial features)  
- ✅ **User-friendly & customizable** with minimal coding required  
- ✅ **Supports different classifiers** (XGBoost, SVM, RandomForest, etc.)  
- ✅ **Optimized model selection** using `BayesSearchCV`  

## 📌 How It Works
1. **Prepare Data**: Provide accelerometer data (`x, y, z, timestamps`) with optional self-reports  
2. **Train a Model**: Extract features and train a classifier on labeled movement data  
3. **Classify Movement**: Apply the trained model to classify new movement data  
4. **Analyze Results**: View emotion predictions and progression over time  

## 📄 **Link to Documentation**
[View Documentation](https://mininato.github.io/UserTesting/)

---
title: EmotionRecognitionPipeline
emoji: 👀
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
