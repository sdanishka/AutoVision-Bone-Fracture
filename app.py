import streamlit as st
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(
    layout="wide",
    page_title='Human disease detection',
    page_icon='🦴',
)
st.title('AutoVision Bone Fracture')
st.write("Provide quick and accurate predictions bone fractures.")

# options = ["Select One Disease", "pneumonia", "maleria", "bone fracture"]

# selected_option = st.selectbox("Select One Disease:", options)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# PNEUMONIA_CLASSES = ['Normal', 'Pneumonia']
# MALERIA_CLASSES = ['Parasite', 'Normal']
BONE_FRACTURE_CLASSES = ['fractured', 'No fracture']

# PNEUMONIA_MODEL = tf.keras.models.load_model('./pneumonia.h5')
# MALERIA_MODEL = tf.keras.models.load_model('./maleria.h5')
BONE_MODEL = tf.keras.models.load_model('./bonefracture.h5')


def bone_fracture():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)

        arr = img_to_array(image)
        arr = cv2.resize(arr, (150, 150))
        arr = arr.reshape(-1, 150, 150, 3)
        arr = arr / 255

        prediction = BONE_MODEL.predict([arr])
        confidence_level = round(prediction.max(), 2)
        predicted_class = BONE_FRACTURE_CLASSES[prediction.argmax()]
        st.write(f'Predicted Result : {predicted_class}  and Confidence Level : {confidence_level}')


# def pneumonia():
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", width=250)
#
#         arr = img_to_array(image)
#         arr = cv2.resize(arr, (100, 100))
#         arr = arr.reshape(-1, 100, 100, 3)
#         arr = arr / 255
#
#         prediction = PNEUMONIA_MODEL.predict([arr])
#
#         confidence_level = round(prediction.max(), 2)
#         predicted_class = PNEUMONIA_CLASSES[prediction.argmax()]
#         st.write(f'Predicted Result : {predicted_class}  and Confidence Level : {confidence_level}')
#
#
# def maleria():
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", width=250)
#
#         arr = img_to_array(image)
#         arr = cv2.resize(arr, (224, 224))
#         # arr = arr.reshape(-1, 224,224,3)
#         arr = arr / 255
#         arr = np.expand_dims(arr, axis=0)
#
#         prediction = MALERIA_MODEL.predict(arr)
#
#         confidence_level = round(prediction.max(), 2)
#         predicted_class = MALERIA_CLASSES[prediction.argmax()]
#         st.write(f'Predicted Result : {predicted_class}  and Confidence Level : {confidence_level}')


if __name__ == "__main__":
    bone_fracture()
    # if selected_option == 'pneumonia':
    #     pneumonia()
    # elif selected_option == 'maleria':
    #     maleria()
    # elif selected_option == 'bone fracture':
    #     bone_fracture()
    # elif selected_option == 'Select One Disease':
    #     pass
    # else:
    #     st.write("Something went wrong")
