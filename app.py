import os
from PIL import Image
import streamlit as st
from predictions import predict

# Global variables
project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(project_folder, 'images')

# Streamlit app
st.title("Bone Fracture Detection")

st.markdown("Bone fracture detection system. Upload an X-ray image for fracture detection.")

# Create a two-column layout with a wider right column
col1, col2 = st.columns([3, 1])

# File uploader in the left column
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Predict button
        if st.button('Predict'):
            # Save the uploaded file temporarily in memory
            with open('temp_image.jpg', 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Perform prediction
            bone_type_result = predict('temp_image.jpg')
            result = predict('temp_image.jpg', bone_type_result)

            st.write(f"Prediction Result: {'Fractured' if result == 'fractured' else 'Normal'}")
            st.write(f"Bone Type: {bone_type_result}")

            # Save result
            if st.button('Save Result'):
                screenshot_path = os.path.join(project_folder, 'PredictResults', 'result_temp_image.jpg')
                img.save(screenshot_path)
                st.write(f"Result saved at {screenshot_path}")

# Display rules image in the right column with fixed width
with col2:
    rules_img = Image.open(os.path.join(folder_path, 'rules.jpeg'))
    st.image(rules_img, caption='Rules', use_column_width=True)
