import streamlit as st
import time

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

available_diseases = [
    "Pneumonia Prediction (from X-Ray)",
    "Tuberculosis Prediction (from X-Ray)",
    "Skin Cancer Prediction (from Static Image)"
]

diseases_model_accuracy = {
    "Pneumonia Prediction (from X-Ray)" : "Current Pneumonia model accuracy is 88-90%",
    "Tuberculosis Prediction (from X-Ray)" : "Missing model accuracy information",
    "Skin Cancer Prediction (from Static Image)" : "Missing model accuracy information"
}


st.set_page_config(
    page_title="Medical Diagnosis Platform",
    layout="wide",
    page_icon=":smiley:",
    initial_sidebar_state="auto",
    menu_items=None
)

model_path = r"Medical-Diagnosis-Platform/Models/Pneumonia_Keras_Model/Pneumonia_Keras_Model.h5"
model= load_model(model_path)

st.title("Medical Diagnosis Platform")

st.sidebar.info("Home")
st.sidebar.warning("Available Models")
st.sidebar.error("About")
st.sidebar.success("Contact Us")


col1, col2 = st.columns([2, 1])

selected_model = col1.empty().selectbox("Select a Task/Model", available_diseases)

col2.text("")
col2.empty().button("More information on selected task/model")

if selected_model != None:
    st.warning(diseases_model_accuracy[selected_model])

st.write("#")
output_text = st.empty()

st.write("#\n#")

col3, col4 = st.columns([2.5, 1])


file_status = col3.empty()
file_status.error("No file uploaded")

buffer = col4.file_uploader("Upload test file here for disease prediction")

file_status_2 = col4.empty()
temp_file = NamedTemporaryFile(delete=False)
img = None

flag = 0

if buffer:
    temp_file.write(buffer.getvalue())
    
    try:
      st.write(image.load_img(temp_file.name))
    except:
      file_status.error("Oops... this does seem to be an image file")
      flag = 1


def temp(output):
  file_status_2.success("Your prediction are reading... Have a look above")
  if output[0][0] >= 0.5:
      confidence = round((output[0][0])*100, 2)
      output_text.error(f"This seems to be the case of pneumonia... I am {confidence}% sure")
  else:
      confidence = round((1-output[0][0])*100, 2)
      output_text.success(f"This seems to be a normal case... I am {confidence}% sure")


if not (buffer == None) and (flag == 0):
    file_status.image(buffer)
    file_status_2.warning("Your file is uploaded... Please wait while we predict the output")
    img = image.load_img(temp_file.name, target_size=(500, 500), color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    temp(output)
