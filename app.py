# Creation of Streamlit Application:

# Importing the Libraries 
import pickle as pkl
import sklearn 
import streamlit as st
import numpy as np

# Streamlit application creation:

st.title("INTRODUCTION TO STREAMLIT")

# loading the saved models:

option = st.selectbox("Which Model would you like to use: ",("Logistic Regression","Support Vector Machine","K-Nearest Neighbors","Gaussian Naive Bayes","Decision Tree"))
st.write("Using: ",option)
if option == "Logistic Regression":
    model_LR = pkl.load(open("model_LR.pkl",'rb'))
    st.write(option," Model loaded successfully")
    # Taking inputs from the users:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        SepalLength = st.number_input("Sepal Length(in cms): ")
    with col2:
        SepalWidth = st.number_input("Sepal Width(in cms): ")
    with col3:
        PetalLength = st.number_input("Petal Length(in cms): ")
    with col4:
        PetalWidth = st.number_input("Petal Width(in cms): ")

    st.warning("Check the inputs again...")
    st.button("Submit")
    
    # Reshaping the input data:
    input_data = (SepalLength,SepalWidth,PetalLength,PetalWidth)
    # Changing the inputData to numpy array:
    input_Data_as_numpy_array = np.asarray(input_data)

    # Reshaping the array as we are predicting for one instance:
    input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

    # Prediction:
    iris_species = model_LR.predict(input_Data_Reshaped)
    if iris_species[0] == "Iris-setosa":
        st.success("It is iris-setosa flower")
        st.image("Iris_setosa.jpg")
    if iris_species[0] == "Iris-virginica":
        st.success("It is iris-virginica flower")
        st.image("Iris_virginica.jpg")
    if iris_species[0] == "Iris-versicolor":
        st.success("It is iris-versicolor flower")
        st.image("iris-versicolor.jpg")

elif option == "Support Vector Machine":
    model_SVM = pkl.load(open("model_SVM.pkl",'rb'))
    st.write(option," Model loaded successfully")
    # Taking inputs from the users:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        SepalLength = st.number_input("Sepal Length(in cms): ")
    with col2:
        SepalWidth = st.number_input("Sepal Width(in cms): ")
    with col3:
        PetalLength = st.number_input("Petal Length(in cms): ")
    with col4:
        PetalWidth = st.number_input("Petal Width(in cms): ")

    st.warning("Check the inputs again...")
    st.button("Submit")

    # Reshaping the input data:
    input_data = (SepalLength,SepalWidth,PetalLength,PetalWidth)
    # Changing the inputData to numpy array:
    input_Data_as_numpy_array = np.asarray(input_data)

    # Reshaping the array as we are predicting for one instance:
    input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

    # Prediction:
    iris_species = model_SVM.predict(input_Data_Reshaped)
    if iris_species[0] == "Iris-setosa":
        st.success("It is iris-setosa flower")
        st.image("Iris_setosa.jpg")
    if iris_species[0] == "Iris-virginica":
        st.success("It is iris-virginica flower")
        st.image("Iris_virginica.jpg")
    if iris_species[0] == "Iris-versicolor":
        st.success("It is iris-versicolor flower")
        st.image("iris-versicolor.jpg")
elif option == "K-Nearest Neighbors":
    model_KNN = pkl.load(open("model_KNN.pkl",'rb'))
    st.write(option," Model loaded successfully")
    # Taking inputs from the users:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        SepalLength = st.number_input("Sepal Length(in cms): ")
    with col2:
        SepalWidth = st.number_input("Sepal Width(in cms): ")
    with col3:
        PetalLength = st.number_input("Petal Length(in cms): ")
    with col4:
        PetalWidth = st.number_input("Petal Width(in cms): ")

    st.warning("Check the inputs again...")
    st.button("Submit")

    # Reshaping the input data:
    input_data = (SepalLength,SepalWidth,PetalLength,PetalWidth)
    # Changing the inputData to numpy array:
    input_Data_as_numpy_array = np.asarray(input_data)

    # Reshaping the array as we are predicting for one instance:
    input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

    # Prediction:
    iris_species = model_KNN.predict(input_Data_Reshaped)
    if iris_species[0] == "Iris-setosa":
        st.success("It is iris-setosa flower")
        st.image("Iris_setosa.jpg")
    if iris_species[0] == "Iris-virginica":
        st.success("It is iris-virginica flower")
        st.image("Iris_virginica.jpg")
    if iris_species[0] == "Iris-versicolor":
        st.success("It is iris-versicolor flower")
        st.image("iris-versicolor.jpg")

elif option == "Gaussian Naive Bayes":
    model_GNB = pkl.load(open("model_GNB.pkl",'rb'))
    st.write(option," Model loaded successfully")
    # Taking inputs from the users:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        SepalLength = st.number_input("Sepal Length(in cms): ")
    with col2:
        SepalWidth = st.number_input("Sepal Width(in cms): ")
    with col3:
        PetalLength = st.number_input("Petal Length(in cms): ")
    with col4:
        PetalWidth = st.number_input("Petal Width(in cms): ")

    st.warning("Check the inputs again...")
    st.button("Submit")

    # Reshaping the input data:
    input_data = (SepalLength,SepalWidth,PetalLength,PetalWidth)
    # Changing the inputData to numpy array:
    input_Data_as_numpy_array = np.asarray(input_data)

    # Reshaping the array as we are predicting for one instance:
    input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

    # Prediction:
    iris_species = model_GNB.predict(input_Data_Reshaped)
    if iris_species[0] == "Iris-setosa":
        st.success("It is iris-setosa flower")
        st.image("Iris_setosa.jpg")
    if iris_species[0] == "Iris-virginica":
        st.success("It is iris-virginica flower")
        st.image("Iris_virginica.jpg")
    if iris_species[0] == "Iris-versicolor":
        st.success("It is iris-versicolor flower")
        st.image("iris-versicolor.jpg")
else:
    model_DT = pkl.load(open("model_DT.pkl","rb"))
    st.write(option," Model loaded successfully")
    # Taking inputs from the users:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        SepalLength = st.number_input("Sepal Length(in cms): ")
    with col2:
        SepalWidth = st.number_input("Sepal Width(in cms): ")
    with col3:
        PetalLength = st.number_input("Petal Length(in cms): ")
    with col4:
        PetalWidth = st.number_input("Petal Width(in cms): ")

    st.warning("Check the inputs again...")
    st.button("Submit")

    # Reshaping the input data:
    input_data = (SepalLength,SepalWidth,PetalLength,PetalWidth)
    # Changing the inputData to numpy array:
    input_Data_as_numpy_array = np.asarray(input_data)

    # Reshaping the array as we are predicting for one instance:
    input_Data_Reshaped = input_Data_as_numpy_array.reshape(1,-1)

    # Prediction:
    iris_species = model_DT.predict(input_Data_Reshaped)
    if iris_species[0] == "Iris-setosa":
        st.success("It is iris-setosa flower")
        st.image("Iris_setosa.jpg")
    if iris_species[0] == "Iris-virginica":
        st.success("It is iris-virginica flower")
        st.image("Iris_virginica.jpg")
    if iris_species[0] == "Iris-versicolor":
        st.success("It is iris-versicolor flower")
        st.image("iris-versicolor.jpg")

