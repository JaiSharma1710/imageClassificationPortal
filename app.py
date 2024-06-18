import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor, pipeline
from PIL import Image
import requests
import torch


def predict(image):
    result = []
    feature_extractor = ViTImageProcessor.from_pretrained(
        'sharmajai901/UL_base_classification')
    model = ViTForImageClassification.from_pretrained(
        'sharmajai901/UL_base_classification')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get the top two predicted probabilities and their corresponding class indices
    top_two_probabilities, top_two_indices = torch.topk(
        probabilities, k=2, dim=-1)

    # Print the top two predicted classes and their probabilities
    for prob, idx in zip(top_two_probabilities[0], top_two_indices[0]):
        result.append(
            f"Predicted Label: {model.config.id2label[idx.item()]} ({(prob.item()*100):.2f}%)")
    return result


def objects(image):
    result = []
    pipe = pipeline("image-segmentation",
                    model="facebook/mask2former-swin-large-ade-semantic")
    segments = pipe(image)
    for ele in segments:
        result.append(f"{ele['label']} : {(ele['score']*100):.2f}%")
    return result


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
        predicted_label = predict(image)
        segments = objects(image)
    st.success(predicted_label[0])
    st.success(predicted_label[1])
    st.write('What we see :')
    for ele in segments:
        st.write(ele)
