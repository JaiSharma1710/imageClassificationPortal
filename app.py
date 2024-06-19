import streamlit as st
from transformers import pipeline
from PIL import Image
import requests


def base_prediction(image):
    base_prediction_pipe = pipeline(
        "image-classification", model="sharmajai901/UL_base_classification")
    base_prediction = base_prediction_pipe(image)
    return base_prediction[0]


def sub_prediction(image, base_label):
    if base_label == 'interior':
        print('interior')
    elif base_label == 'exterior':
        exterior_pipe = pipeline(
            "image-classification", model="sharmajai901/UL_exterior_classification")
        exterior_pipe_result = exterior_pipe(image)
        return exterior_pipe_result[0]
    elif base_label == 'bedrooms':
        bedrooms_pipe = pipeline(
            "image-classification", model="sharmajai901/UL_bedroom_classification")
        bedrooms_pipe_result = bedrooms_pipe(image)
        return bedrooms_pipe_result[0]
    elif base_label == 'others':
        return {'label': 'others', 'score': 0.1}
    elif base_label == 'floorPlans':
        return {'label': 'floorPlans', 'score': 0.1}


def is_image_url(url):
    image_extensions = [".jpg", ".jpeg", ".png"]
    for ext in image_extensions:
        if url.lower().endswith(ext):
            return True
    return False


# Streamlit app
st.title("Image Classification")

URL = st.text_input("Image URL")

if st.button("Classify"):
    if not URL:
        st.error('no URL present')

    if not is_image_url(URL):
        st.error('only jpg, jpeg, png format allowed')

    image = Image.open(requests.get(URL, stream=True).raw)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        base_predicted_result = base_prediction(image)
        sub_prediction_result = sub_prediction(
            image=image, base_label=base_predicted_result['label'])

    st.success(
        f"Base Category : {base_predicted_result['label']} ( {(base_predicted_result['score']*100):.2f}% )")

    if sub_prediction_result:
        st.success(
            f"Sub Category : {sub_prediction_result['label']} ( {(sub_prediction_result['score']*100):.2f}% )")
    else:
        st.error('not a valid category')
