import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



st.title('Iris Classifier')
st.write("This app uses 6 inputs to predict the Variety of Iris using "
         "a model built on the Palmer's Iris's dataset. Use the form below"
         " to get started!")

iris_file = st.file_uploader('Upload your own Iris data')

if iris_file is None:
    rf_pickle = open('/mount/src/streamlit_66056040_iris/Iris_ml/random_forest_iris.pickle', 'rb')
    map_pickle = open('/mount/src/streamlit_66056040_iris/Iris_ml/output_iris.pickle', 'rb')

    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)


else:
    iris_df = pd.read_csv(iris_file)
    iris_df = iris_df.dropna()