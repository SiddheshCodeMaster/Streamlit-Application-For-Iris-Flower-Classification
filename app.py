# Importing important Libraries and Modules:
import pickle as pkl
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu

# Loading the saved models:

saved_model_LR = pkl.load(open("model_LR.pkl",'rb'))
saved_model_SVM = pkl.load(open("model_SVM.pkl",'rb'))
saved_model_KNN = pkl.load(open("model_KNN.pkl",'rb'))
saved_model_GNB = pkl.load(open("model_GNB.pkl",'rb'))
saved_model_DT = pkl.load(open("model_DT.pkl",'rb'))

# Sidebar for Navigation:
with st.sidebar:
    selected = option_menu('Iris Flower Prediction System with Different ML Models',
                           ['Logistic Regression','Support Vector Machine','K-Nearest Neighbors','Gaussian Naive Bayes Classifier',
                            'Decision Tree'], default_index=0)
    
# Logistic Regression Page:
if selected == 'Logistic Regression':
    # Page title:
    st.title('Iris-Flower Classification using Logistic Regression')

    # Getting the input from the user:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Sepal_length = st.text_input("Enter the Sepal Lenght(in cms):")
    with col2:
        Sepal_width = st.text_input("Enter the Sepal Width(in cms):")
    with col3:
        petal_length = st.text_input("Enter the Petal Length(in cms):")
    with col4:
        petal_width = st.text_input("Enter the Petal Width(in cms):")        

    # Creating a button for classifcation:
    if st.button("Submit for Classification"):
        input_data = (Sepal_length, Sepal_width, petal_length, petal_width)
        # Changing the inputData to numpy array:
        input_Data_as_numpy_array = np.asarray(input_data)

        # Reshaping the array as we are predicting for one instance:
        input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

        predict_iris = saved_model_LR.predict(input_Data_Reshaped)

        if predict_iris[0] == 'Iris-setosa':
            st.write('The flower is Iris-setosa')
            st.image('Iris_setosa.jpg')
        elif predict_iris[0] == 'Iris-virginica':
            st.write('The flower is Iris-virginica')
            st.image('Iris_virginica.jpg')
        elif predict_iris[0] == 'Iris-versicolor':
            st.write('The flower is Iris-versicolor')
            st.image('iris-versicolor.jpg')
        else:
            st.warning('Please check the inputs again...')

# Support Vector Machine Page:
if selected == 'Support Vector Machine':
    # Page title:
    st.title('Iris-Flower Classification using Support Vector Machine')

    # Getting the input from the user:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Sepal_length = st.text_input("Enter the Sepal Lenght(in cms):")
    with col2:
        Sepal_width = st.text_input("Enter the Sepal Width(in cms):")
    with col3:
        petal_length = st.text_input("Enter the Petal Lenght(in cms):")
    with col4:
        petal_width = st.text_input("Enter the Petal Width(in cms):")        

    # Creating a button for classifcation:
    if st.button("Submit for Classification"):
        input_data = (Sepal_length, Sepal_width, petal_length, petal_width)
        # Changing the inputData to numpy array:
        input_Data_as_numpy_array = np.asarray(input_data)

        # Reshaping the array as we are predicting for one instance:
        input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

        predict_iris = saved_model_SVM.predict(input_Data_Reshaped)

        if predict_iris[0] == 'Iris-setosa':
            st.write('The flower is Iris-setosa')
            st.image('Iris_setosa.jpg')
        elif predict_iris[0] == 'Iris-virginica':
            st.write('The flower is Iris-virginica')
            st.image('Iris_virginica.jpg')
        elif predict_iris[0] == 'Iris-versicolor':
            st.write('The flower is Iris-versicolor')
            st.image('iris-versicolor.jpg')
        else:
            st.warning('Please check the inputs again...')

# K-Nearest Neighbors Page:
if selected == 'K-Nearest Neighbors':
    # Page title:
    st.title('Iris-Flower Classification using K-Nearest Neighbors')

    # Getting the input from the user:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Sepal_length = st.text_input("Enter the Sepal Lenght(in cms):")
    with col2:
        Sepal_width = st.text_input("Enter the Sepal Width(in cms):")
    with col3:
        petal_length = st.text_input("Enter the Petal Lenght(in cms):")
    with col4:
        petal_width = st.text_input("Enter the Petal Width(in cms):")        

    # Creating a button for classifcation:
    if st.button("Submit for Classification"):
        input_data = (Sepal_length, Sepal_width, petal_length, petal_width)
        # Changing the inputData to numpy array:
        input_Data_as_numpy_array = np.asarray(input_data)

        # Reshaping the array as we are predicting for one instance:
        input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

        predict_iris = saved_model_KNN.predict(input_Data_Reshaped)

        if predict_iris[0] == 'Iris-setosa':
            st.write('The flower is Iris-setosa')
            st.image('Iris_setosa.jpg')
        elif predict_iris[0] == 'Iris-virginica':
            st.write('The flower is Iris-virginica')
            st.image('Iris_virginica.jpg')
        elif predict_iris[0] == 'Iris-versicolor':
            st.write('The flower is Iris-versicolor')
            st.image('iris-versicolor.jpg')
        else:
            st.warning('Please check the inputs again...')

# Gaussian Naive Bayes Classifier Page:
if selected == 'Gaussian Naive Bayes Classifier':
    # Page title:
    st.title('Iris-Flower Classification using Gaussian Naive Bayes Classifier')

    # Getting the input from the user:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Sepal_length = st.text_input("Enter the Sepal Lenght(in cms):")
    with col2:
        Sepal_width = st.text_input("Enter the Sepal Width(in cms):")
    with col3:
        petal_length = st.text_input("Enter the Petal Lenght(in cms):")
    with col4:
        petal_width = st.text_input("Enter the Petal Width(in cms):")        

    # Creating a button for classifcation:
    if st.button("Submit for Classification"):
        input_data = (Sepal_length, Sepal_width, petal_length, petal_width)
        # Changing the inputData to numpy array:
        input_Data_as_numpy_array = np.asarray(input_data)

        # Reshaping the array as we are predicting for one instance:
        input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

        predict_iris = saved_model_GNB.predict(input_Data_Reshaped)

        if predict_iris[0] == 'Iris-setosa':
            st.write('The flower is Iris-setosa')
            st.image('Iris_setosa.jpg')
        elif predict_iris[0] == 'Iris-virginica':
            st.write('The flower is Iris-virginica')
            st.image('Iris_virginica.jpg')
        elif predict_iris[0] == 'Iris-versicolor':
            st.write('The flower is Iris-versicolor')
            st.image('iris-versicolor.jpg')
        else:
            st.warning('Please check the inputs again...')

# Decision Tree Page:
if selected == 'Decision Tree':
    # Page title:
    st.title('Iris-Flower Classification using Decision Tree')

    # Getting the input from the user:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Sepal_length = st.text_input("Enter the Sepal Lenght(in cms):")
    with col2:
        Sepal_width = st.text_input("Enter the Sepal Width(in cms):")
    with col3:
        petal_length = st.text_input("Enter the Petal Lenght(in cms):")
    with col4:
        petal_width = st.text_input("Enter the Petal Width(in cms):")        

    # Creating a button for classifcation:
    if st.button("Submit for Classification"):
        input_data = (Sepal_length, Sepal_width, petal_length, petal_width)
        # Changing the inputData to numpy array:
        input_Data_as_numpy_array = np.asarray(input_data)

        # Reshaping the array as we are predicting for one instance:
        input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

        predict_iris = saved_model_DT.predict(input_Data_Reshaped)

        if predict_iris[0] == 'Iris-setosa':
            st.write('The flower is Iris-setosa')
            st.image('Iris_setosa.jpg')
        elif predict_iris[0] == 'Iris-virginica':
            st.write('The flower is Iris-virginica')
            st.image('Iris_virginica.jpg')
        elif predict_iris[0] == 'Iris-versicolor':
            st.write('The flower is Iris-versicolor')
            st.image('iris-versicolor.jpg')
        else:
            st.warning('Please check the inputs again...')