import tkinter as tk
from tkinter import filedialog, Label, Button, Text
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pickle

# Initialize main Tkinter window
main = tk.Tk()
main.title("Diabetic Retinopathy Prediction")
main.geometry("1000x650")
main.config(bg='light coral')

global dataset_path, cnn_model, train_batches, valid_batches, test_batches

def load_dataset():
    global dataset_path
    dataset_path = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', tk.END)
    text.insert(tk.END, f"Dataset loaded from: {dataset_path}\n")

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

def process_images():
    global train_batches, valid_batches, test_batches
    if not dataset_path:
        text.insert(tk.END, "Error: Please upload a dataset first!\n")
        return

    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    if not all(map(os.path.exists, [train_path, valid_path, test_path])):
        text.insert(tk.END, "Error: Invalid dataset structure!\n")
        return
    
    datagen = ImageDataGenerator(rescale=1./255)
    train_batches = datagen.flow_from_directory(train_path, target_size=(224,224), batch_size=10, class_mode='categorical')
    valid_batches = datagen.flow_from_directory(valid_path, target_size=(224,224), batch_size=10, class_mode='categorical')
    test_batches = datagen.flow_from_directory(test_path, target_size=(224,224), batch_size=10, class_mode='categorical')
    
    text.insert(tk.END, "Image preprocessing completed.\n")

def run_cnn():
    global cnn_model, train_batches, valid_batches
    if 'train_batches' not in globals():
        text.insert(tk.END, "Error: Please preprocess images first!\n")
        return
    
    vgg16_model = VGG16(include_top=False, input_shape=(224, 224, 3))
    cnn_model = Sequential([
        vgg16_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])
    cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint('diabetic_retinopathy.keras', monitor='val_accuracy', save_best_only=True)
    history = cnn_model.fit(train_batches, validation_data=valid_batches, epochs=30, verbose=2, callbacks=[checkpoint])
    
    os.makedirs('model', exist_ok=True)
    with open('model/history.pckl', 'wb') as f:
        pickle.dump(history.history, f)
    
    text.insert(tk.END, "CNN training completed.\n")

def predict(): 
    global cnn_model
    model_path = "diabetic_retinopathy.keras"
    
    if not os.path.exists(model_path):
        text.insert(tk.END, "Error: Model not trained! Train the model first.\n")
        return
    
    cnn_model = load_model(model_path)  # Load saved model
    
    filename = filedialog.askopenfilename(initialdir="testImages", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not filename:
        return
    
    img = Image.open(filename)
    processed_image = preprocess_image(img)
    prediction = cnn_model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    result = "Diabetic Retinopathy Detected" if predicted_class == 0 else "No Diabetic Retinopathy Detected"
    text.insert(tk.END, f"Prediction: {result}\n")

    img = cv2.imread(filename)
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label_img = Label(main, image=img)
    label_img.image = img
    label_img.place(x=700, y=100)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def plot_graph():
    history_path = 'model/history.pckl'
    if not os.path.exists(history_path):
        text.insert(tk.END, "Error: No training history found! Train the model first.\n")
        return
    
    with open(history_path, 'rb') as f:
        data = pickle.load(f)
    
    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.plot(loss, 'r', label='Loss')
    plt.plot(accuracy, 'g', label='Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.title('CNN Accuracy & Loss Graph')
    plt.show()

# UI Components
font = ('times', 16, 'bold')
title = Label(main, text='Diabetic Retinopathy Prediction', bg='lavender blush', fg='DarkOrchid1', font=font, height=3, width=120)
title.pack()

font1 = ('times', 13, 'bold')
Button(main, text="Upload Dataset", command=load_dataset, font=font1).place(x=10, y=100)
Button(main, text="Preprocess Images", command=process_images, font=font1).place(x=200, y=100)
Button(main, text="Run CNN", command=run_cnn, font=font1).place(x=400, y=100)
Button(main, text="Predict Image", command=predict, font=font1).place(x=10, y=200)
Button(main, text="Accuracy & Loss Graph", command=plot_graph, font=font1).place(x=200, y=200)

text = Text(main, height=20, width=120)
text.place(x=10, y=300)

main.mainloop()  
