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

iris_file = st.file_uploader('upload')

if iris_file is None:
    rf_pickle = open('stl_app/06_iris_ml/random_forest_iris.pickle', 'rb')
    map_pickle = open('stl_app/06_iris_ml/output_iris.pickle', 'rb')

    rfc = pickle.load(rf_pickle)
    iris_df = pd.read_csv('stl_app/06_iris_ml/iris.csv')
    unique_penguin_mapping = pickle.load(map_pickle)

    rf_pickle.close()
else:
    iris_df = pd.read_csv(iris_file)
    iris_df = iris_df.dropna()

    output = iris_df['variety']
    features = iris_df[['sepal.length',
           'sepal.width',
           'petal.length',
           'petal.width']]

    features = pd.get_dummies(features)

    output, unique_penguin_mapping = pd.factorize(output)

    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=.8)

    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)

    score = round(accuracy_score(y_pred, y_test), 2)

    st.write('We trained a Random Forest model on these data,'
             ' it has a score of {}! Use the '
             'inputs below to try out the model.'.format(score))

with st.form('user_inputs'):
    sepal_length = st.number_input(
        'Sepal Length', min_value=0.0, max_value=12.0, value=10.0)
    sepal_width = st.number_input(
        'Sepal Width', min_value=0.0, max_value=12.0, value=10.0)
    petal_length = st.number_input(
        'Petal Length', min_value=0.0, max_value=12.0, value=10.0)
    petal_width = st.number_input(
        'Petal Width', min_value=0.0, max_value=12.0, value=10.0)
    st.form_submit_button()


new_prediction = rfc.predict([[sepal_length, sepal_width, petal_length,
                               petal_width]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write('We predict your Iris is of the {} species'.format(prediction_species))



st.title("Iris ")
st.markdown('สร้าง `scatter plot` แสดงผลข้อมูล **Palmer\'s Penguins** กัน แบบเดียวกับ **Iris dataset**')

choices = ['sepal.length',
           'sepal.width',
           'petal.length',
           'petal.width']

selected_x_var = st.selectbox('เลือก แกน x', (choices))
selected_y_var = st.selectbox('เลือก แกน y', (choices))


st.subheader('ข้อมูลตัวอย่าง')
st.write(iris_df)

st.subheader('แสดงผลข้อมูล')
sns.set_style('darkgrid')
markers = {"Setosa": "v", "Versicolor": "s", "Virginica": 'o'}

fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_df,
                     x=selected_x_var, y=selected_y_var,
                     hue='variety', markers=markers, style='variety')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Iris Data")
st.pyplot(fig)