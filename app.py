import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Model
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the trained model (.keras model)
model = tf.keras.models.load_model('/content/drive/MyDrive/NIH Dataset/DenseNet121_val_loss (1).keras')

# Class names (replace with the actual class names for your 14-class problem)
label_dict = {
    0: 'Atelectasis', 1: 'Consolidation', 2: 'Infiltration', 3: 'Pneumothorax', 
    4: 'Edema', 5: 'Emphysema', 6: 'Fibrosis', 7: 'Effusion', 
    8: 'Pneumonia', 9: 'Pleural_Thickening', 10: 'Cardiomegaly', 
    11: 'Nodule', 12: 'Mass', 13: 'Hernia'
}

# Function to preprocess the image before feeding it to the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the required input size (adjust size if necessary)
    image = np.array(image)             # Convert image to a NumPy array
    image = image / 255.0               # Normalize pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return image

# Function to create heatmap
def create_heatmap(img_path, model):
    # Load the DenseNet121 model
    last_conv_layer = model.get_layer('conv5_block16_concat')
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # Load and preprocess the image
    image = load_img(img_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)

    # Extract features from the image
    features = last_conv_layer_model.predict(preprocessed_image)

    # Calculate the weights for each channel in the feature map
    weights = np.mean(features, axis=(1, 2))
    weights = np.maximum(weights, 0)

    # Resize the image and the attention map for visualization
    upsampled_image = np.uint8(255 * (image_array[0] + 1) / 2)  # Rescale the image to 0-255
    upsampled_attention = cv2.resize(weights, (upsampled_image.shape[1], upsampled_image.shape[0]))
    upsampled_attention = np.uint8(255 * upsampled_attention)  # Convert weights to 0-255
    heatmap = cv2.applyColorMap(upsampled_attention, cv2.COLORMAP_JET)

    # Overlay the heatmap on the image
    alpha = 0.4  # Adjust for better visibility
    output_image = cv2.addWeighted(upsampled_image, 1 - alpha, heatmap, alpha, 0)

    # Convert the output image to a format that can be displayed in Streamlit
    return output_image

# Streamlit app layout
st.title("X-ray Image Classification App")
st.write("Upload an X-ray image and press the 'Predict' button to see the result.")

# File uploader for users to upload an X-ray image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess the image for prediction
    processed_image = preprocess_image(image)

    # Create and display the heatmap
    heatmap_image = create_heatmap(uploaded_file, model)
    st.image(heatmap_image, caption="Heatmap Overlay", use_column_width=True)

    # Add a "Predict" button
    if st.button('Predict'):
        # Make a prediction using the model
        predictions = model.predict(processed_image)

        # Get the index of the class with the highest prediction score
        predicted_class_idx = np.argmax(predictions[0])

        # Get the predicted class name
        predicted_class_name = label_dict[predicted_class_idx]

        # Display the prediction result
        st.write(f"Highest Probability Class: {predicted_class_name}")

        # Loop over the predicted probability array and display each label and its probability
        for i in range(len(predictions[0])):
            disease = label_dict[i]
            probability = predictions[0][i]  # Access the specific float value
            st.write(f'{disease}: {probability:.4f}')
