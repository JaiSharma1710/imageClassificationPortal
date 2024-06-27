import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID

client = Client()
client.set_endpoint('https://cloud.appwrite.io/v1')
client.set_project('6669963100239a2ad4ee')
client.set_key('a8fe71b0ef0f2066855d0b37ccb786d4eb46a186777bd8e9c13d331753cd26ffb79db0beaad8f35d0e7af1ba008593a743f7786c7490642adcd73870dad6c5ba1f0e50c6faecd66b6a945aa2c3d43f320245e5e225c3f3ebd659cf7634a3ccecdde95148e8697a18b174b9f27b40927c7fd39d1998e9cd6b3a1aa1148863ec5c')
databases = Databases(client)


def base_prediction(image):
    base_prediction_pipe = pipeline(
        "image-classification", model="sharmajai901/UL_base_classification")
    base_prediction = base_prediction_pipe(image)
    return base_prediction[0]


def sub_prediction(image, base_label):
    if base_label == 'interior':
        interior_pipe = pipeline(
            "image-classification", model="sharmajai901/UL_interior_classification")
        interior_pipe_result = interior_pipe(image)
        return interior_pipe_result[0]
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
if 'base_prediction' not in st.session_state:
    st.session_state['base_prediction'] = ''

if 'sub_prediction' not in st.session_state:
    st.session_state['sub_prediction'] = ''


def submitResponse(result, main, sub, expected_main, expected_sub):
    try:
        if (not URL):
            st.error('No image URL found')
        else:
            if (result == 'Fail'):
                doc = {
                    'shownMainCategory': main,
                    'shownSubCategory': sub,
                    'expectedMainCategory': expected_main,
                    'expectedSubCategory': expected_sub,
                    'result': result,
                    'imageUrl': URL
                }
            else:
                doc = {
                    'shownMainCategory': main,
                    'shownSubCategory': sub,
                    'result': result,
                    'imageUrl': URL
                }

            # Assuming ID.unique() returns a unique identifier for the document
            document_id = ID.unique()

            # Create document in Appwrite database
            databases.create_document(
                database_id='66699660003593cbbad5',
                collection_id='6669967d0017e1f7ffcf',
                document_id=document_id,
                data=doc
            )
            st.success('submitted')
    except Exception as e:
        st.error(f'Error occurred: {str(e)}')


def getOptions(selection):
    sub_category_option = []
    if selection == 'interior':
        sub_category_option = [
            'communal lounge',
            'living area',
            'study area',
            'cinema room',
            'laundry area',
            'swimming pool',
            'gym',
            'building interiors',
            'games area',
            'dining area',
            'reception',
            'bicycle storage',
            'rooftoop area',
            'fitness room',
            'living area & shared kitchen',
            'parking',
            'meeting room',
            'grocery store',
            'storage lockers',
            'entertainment area'
        ]
    elif selection == 'exterior':
        sub_category_option = [
            'outdoor area',
            'street view',
            'building exterior'
        ]
    elif selection == 'bedrooms':
        sub_category_option = ['bedroom', 'kitchen', 'bathroom']
    elif selection == 'others':
        sub_category_option = ['others']
    elif selection == 'floorPlans':
        sub_category_option = ['floorPlans']
    return sub_category_option


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
        st.session_state.base_prediction = base_predicted_result['label']
        st.session_state.sub_prediction = sub_prediction_result['label']

    st.success(
        f"Base Category : {base_predicted_result['label']} ( {(base_predicted_result['score']*100):.2f}% )")

    if sub_prediction_result:
        st.success(
            f"Sub Category : {sub_prediction_result['label']} ( {(sub_prediction_result['score']*100):.2f}% )")
    else:
        st.error('not a valid category')

with st.sidebar:
    st.text_input(placeholder="Main Category",
                  value=st.session_state.base_prediction, disabled=True, label='Main Category')

    st.text_input(placeholder="Sub Category",
                  value=st.session_state.sub_prediction, disabled=True, label='Sub Category')

    isSuccess = st.selectbox(
        "is Success",
        ['Success', 'Fail'],
        index=None,
        placeholder="Select result",
    )

    if isSuccess:
        if isSuccess == 'Fail':
            option_Fail_main = st.selectbox(
                "Expected main category",
                ["interior", "exterior", "bedrooms", "others", 'floorPlans'],
                index=None,
                placeholder="Select expected main category",
            )
            if option_Fail_main:
                option_Fail_sub = st.selectbox(
                    "Expected sub category",
                    getOptions(option_Fail_main),
                    index=None,
                    placeholder="Select expected sub category.",
                )
            if option_Fail_main and option_Fail_sub:
                if st.button('Submit'):
                    submitResponse(
                        'Fail', st.session_state.base_prediction, st.session_state.sub_prediction, option_Fail_main, option_Fail_sub)
        else:
            if st.button('Submit'):
                submitResponse(
                    'Success', st.session_state.base_prediction, st.session_state.sub_prediction, '', '')
