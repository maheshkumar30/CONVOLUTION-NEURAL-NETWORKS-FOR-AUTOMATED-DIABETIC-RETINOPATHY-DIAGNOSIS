import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# Initialize session state
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = None
if 'cnn_model' not in st.session_state:
    st.session_state.cnn_model = None
if 'train_batches' not in st.session_state:
    st.session_state.train_batches = None
if 'valid_batches' not in st.session_state:
    st.session_state.valid_batches = None
if 'test_batches' not in st.session_state:
    st.session_state.test_batches = None

st.title("Diabetic Retinopathy Prediction")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload Dataset", "Train CNN", "Predict Image", "View Accuracy & Loss"])

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

if page == "Upload Dataset":
    dataset_path = st.text_input("Enter Dataset Path")
    if st.button("Load Dataset"):
        if os.path.exists(dataset_path):
            st.session_state.dataset_path = dataset_path
            st.success(f"Dataset loaded from: {dataset_path}")
        else:
            st.error("Invalid dataset path!")

    if st.session_state.dataset_path:
        if st.button("Preprocess Images"):
            train_path = os.path.join(st.session_state.dataset_path, 'train')
            valid_path = os.path.join(st.session_state.dataset_path, 'valid')
            test_path = os.path.join(st.session_state.dataset_path, 'test')

            if not all(map(os.path.exists, [train_path, valid_path, test_path])):
                st.error("Error: Invalid dataset structure!")
            else:
                datagen = ImageDataGenerator(rescale=1./255)
                st.session_state.train_batches = datagen.flow_from_directory(train_path, target_size=(224,224), batch_size=10, class_mode='categorical')
                st.session_state.valid_batches = datagen.flow_from_directory(valid_path, target_size=(224,224), batch_size=10, class_mode='categorical')
                st.session_state.test_batches = datagen.flow_from_directory(test_path, target_size=(224,224), batch_size=10, class_mode='categorical')
                st.success("Image preprocessing completed.")

elif page == "Train CNN":
    if st.button("Run CNN"):
        if st.session_state.train_batches is None:
            st.error("Error: Please preprocess images first!")
        else:
            vgg16_model = VGG16(include_top=False, input_shape=(224, 224, 3))
            cnn_model = Sequential([
                vgg16_model,
                Flatten(),
                Dense(256, activation='relu'),
                Dense(2, activation='softmax')
            ])
            cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            
            checkpoint = ModelCheckpoint('diabetic_retinopathy_best.keras', monitor='val_accuracy', save_best_only=True)
            history = cnn_model.fit(st.session_state.train_batches, validation_data=st.session_state.valid_batches, epochs=12, verbose=2, callbacks=[checkpoint])
            
            os.makedirs('model', exist_ok=True)
            with open('model/history.pckl', 'wb') as f:
                pickle.dump(history.history, f)
            
            st.session_state.cnn_model = cnn_model
            st.success("CNN training completed.")

elif page == "Predict Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            if st.session_state.cnn_model is None:
                st.session_state.cnn_model = load_model("diabetic_retinopathy_best.keras")
            processed_image = preprocess_image(image)
            prediction = st.session_state.cnn_model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            result = "Diabetic Retinopathy Detected" if predicted_class == 0 else "No Diabetic Retinopathy Detected"
            st.success(f"Prediction: {result}")

elif page == "View Accuracy & Loss":
    history_path = 'model/history.pckl'
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            data = pickle.load(f)
        
        fig, ax = plt.subplots()
        ax.plot(data['loss'], 'r', label='Loss')
        ax.plot(data['accuracy'], 'g', label='Accuracy')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy/Loss')
        ax.legend()
        ax.set_title('CNN Accuracy & Loss Graph')
        st.pyplot(fig)
    else:
        st.error("Error: No training history found! Train the model first.")