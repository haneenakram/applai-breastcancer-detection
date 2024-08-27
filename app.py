import streamlit as st
import requests
import joblib
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(
    page_title='Breast Cancer Classifier',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'  # Collapsed sidebar
)


def load_lottie(url):  # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


model = joblib.load(open("breastCancerClasifier", 'rb'))

# Load the scaler
scaler = joblib.load('scaler.joblib')


def predict(radius_mean, concavity_mean, concavity_worst,	radius_se,	compactness_mean,	compactness_worst,	texture_mean,	smoothness_worst,	smoothness_mean, concavity_se,	concave_points_se,	symmetry_worst,	fractal_dimension_mean):

    # Create the feature array
    features = np.array([radius_mean, concavity_mean, concavity_worst, radius_se, compactness_mean, compactness_worst, texture_mean,
                        smoothness_worst, smoothness_mean, concavity_se, concave_points_se, symmetry_worst, fractal_dimension_mean]).reshape(1, -1)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)

    # Make the prediction using the scaled features
    prediction = round(model.predict(scaled_features)[0])
    return prediction


with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs", "About", "Contact"],
                         icons=['house', 'kanban',
                                'book', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == 'Home':
    st.write('# Breast Cancer Classifier')
    st.write('---')
    st.subheader('Enter the details to classify the cancer')

    # User input
    radius_mean = st.number_input(
        "Enter the radius_mean: ", min_value=0.00000, step=0.000001, format="%.6f")
    concavity_mean = st.number_input(
        "Enter the concavity_mean: ", min_value=0.00000, step=0.000001, format="%.6f")
    concavity_worst = st.number_input(
        "Enter concavity_worst:", min_value=0.00000, step=0.000001, format="%.6f")
    radius_se = st.number_input(
        "Enter radius_se:", min_value=0.00000, step=0.000001, format="%.6f")
    compactness_mean = st.number_input(
        "Enter compactness_mean:", min_value=0.00000, step=0.000001, format="%.6f")
    compactness_worst = st.number_input(
        "Enter compactness_worst:", min_value=0.00000, step=0.000001, format="%.6f")
    texture_mean = st.number_input(
        "Enter texture_mean:", min_value=0.00000, step=0.000001, format="%.6f")
    smoothness_worst = st.number_input(
        "Enter smoothness_worst:", min_value=0.00000, step=0.000001, format="%.6f")
    smoothness_mean = st.number_input(
        "Enter smoothness_mean:", min_value=0.00000, step=0.000001, format="%.6f")
    concavity_se = st.number_input(
        "Enter concavity_se:", min_value=0.00000, step=0.000001, format="%.6f")
    concave_points_se = st.number_input(
        "Enter concave_points_se:", min_value=0.00000, step=0.000001, format="%.6f")
    symmetry_worst = st.number_input(
        "Enter symmetry_worst:", min_value=0.00000, step=0.000001, format="%.6f")
    fractal_dimension_mean = st.number_input(
        "Enter fractal_dimension_mean:", min_value=0.00000, step=0.000001, format="%.6f")

    # Predict the cluster
    sample_prediction = predict(radius_mean, concavity_mean, concavity_worst,	radius_se,	compactness_mean,	compactness_worst,
                                texture_mean,	smoothness_worst,	smoothness_mean, concavity_se,	concave_points_se,	symmetry_worst,	fractal_dimension_mean)

    if st.button("Predict"):
        print(sample_prediction)
        if sample_prediction == 0:
            st.warning("Predicted Cancer: B")
            st.write("This indicates a Benign Cancer.")
        elif sample_prediction == 1:
            st.success("Predicted Cancer: M")
            st.write("This indicates a Malignant Cancer.")
            # st.balloons()

elif choose == 'About':
    st.write('# About Page')
    st.write('---')
    st.write("🎯💡 Welcome to our Breast Cancer Classification AI project! Our mission is to leverage cutting-edge artificial intelligence to aid in the early detection and accurate classification of breast cancer, ultimately contributing to better patient outcomes and advancing medical research.Our VisionWe envision a world where advanced technology empowers healthcare professionals to make more informed decisions, leading to earlier diagnoses and more effective treatments for breast cancer patients. . Contact us today to learn more. 📞📧")

elif choose == "Contact":
    st.write('# Contact Us')
    st.write('---')
    # set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    with st.form(key='columns_in_form2', clear_on_submit=True):
        st.write('## Please help us improve!')
        Name = st.text_input(label='Please Enter Your Name')
        Email = st.text_input(label='Please Enter Email')
        Message = st.text_input(label='Please Enter Your Message')
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write(
                'Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')

elif choose == 'Graphs':
    st.write('# Breast cancer Classifier Graphs')
    st.write('---')
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### removing features that have high colleration with each others:")
    st.image("heatmap.png")
    st.write("### after dropping the highly collerated columns")
    st.image("after-dropping.png")
    st.write("## Selecting the more important features")
    st.write("### descending order of the most affecting features on the diognose")
    st.image("features-barchart.png")
    st.write("### the most important features for diognose predection")
    st.image("features-boxplot.png")
    st.write("## Models accuracy comparision")
    st.image("models-comparision.png")

    data = pd.read_csv('breast-cancer.csv')
    # Create a DataFrame
    df = pd.DataFrame(data)
