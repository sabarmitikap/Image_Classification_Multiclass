import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import joblib


@st.cache(allow_output_mutation = True)

def load_model():
  model = tf.keras.models.load_model('Model_Flower.hdf5')
  return model
with st.spinner('Load Model...'):
  model = load_model()

st.write("""
         # Flower Image Classification - Final Project 2
         """
         )

file = st.file_uploader("Please Upload Image File..", type = ["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (180, 180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize = (75, 75)    
        #interpolation = cv2.INTER_CUBIC))/255
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please Upload Image File!")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
    image_class = str(predictions[0])
    score = tf.nn.softmax(predictions[0])
    class_names = ['Sunflowers', 'Daisy', 'Tulips', 'Dandelion', 'Roses']
    print("The Result Show Similarities With {}, The Percentage Result Is {:.2f}".format(class_names[np.argmax(score)], 100 * np.max(score)))