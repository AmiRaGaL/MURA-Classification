import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from GradCAM import grad_cam

# Function to load and preprocess the image
def preprocess_image(image):
    #img = image.convert('RGB') 
    img = image.resize((224, 224))  # Resize the image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Convert image to RGB if it's grayscale
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    return img_array

# Function to load the model
def load_classifier_model():
    model = load_model('densenetmodel.keras')
    return model

# Function to make predictions
def predict(image, model):
    image = preprocess_image(image)
    preds = model.predict(image)
    return preds


# Main function to run the Streamlit app
def main():
    st.title('Image Classification App')
    st.write('Upload an image for classification')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)

        model = load_classifier_model()
        preds = predict(image, model)
        Class_labels = ['negative', 'positive']
        predicted_class_index = np.argmax(preds)
        predicted_class = Class_labels[predicted_class_index]

        # Display predictions
        st.write('Predictions:')
        st.write(f"Class: {predicted_class}")
        
        # Generate and display Grad-CAM
        grad_cam_image = grad_cam(preprocess_image(image), model, last_conv_layer_name='conv5_block16_concat', classifier_layer_names=None)
        st.image(grad_cam_image, caption='Grad-CAM', use_column_width=True)

if __name__ == '__main__':
    main()
