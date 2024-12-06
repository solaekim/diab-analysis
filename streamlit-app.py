import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Title and Sidebar Navigation
st.title("Diabetes Data Analysis and Prediction")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Training", "Prediction"])

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Home Page
if page == "Home":
    st.write("""
    Welcome to the Diabetes Analysis and Prediction App!
    Upload your dataset to gain insights and make predictions.
    """)

# EDA Page
if page == "EDA":
    if not uploaded_file:
        st.error("Please upload a CSV file to perform EDA.")
    else:
        st.header("Exploratory Data Analysis")
        
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        st.subheader("Distribution of Outcome")
        try:
            outcome_counts = df['Outcome'].value_counts()
            st.bar_chart(outcome_counts)
        except KeyError:
            st.error("The dataset does not contain an 'Outcome' column.")

        st.subheader("Correlation Heatmap")
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        except ValueError:
            st.error("Unable to compute the correlation heatmap. Check your data.")

        st.subheader("Feature Distributions")
        for column in df.columns[:-1]:  # Exclude 'Outcome'
            try:
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                plt.axvline(df[column].mean(), color='red', linestyle='dashed', linewidth=1)
                st.pyplot(fig)
            except KeyError:
                st.error(f"Column '{column}' not found in the dataset.")

# Model Training Page
if page == "Model Training":
    if not uploaded_file:
        st.error("Please upload a CSV file to train the model.")
    else:
        try:
            X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                    'BMI', 'DiabetesPedigreeFunction', 'Age']]
            y = df['Outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            st.header("Model Performance")
            st.write(f"Accuracy: {model.score(X_test, y_test):.2f}")
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

            st.subheader("Feature Coefficients")
            coefficients = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": np.squeeze(model.coef_)
            }).sort_values(by="Coefficient", ascending=False)
            st.write(coefficients)
            
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during model training: {e}")

# Prediction Page
if page == "Prediction":
    if not uploaded_file:
        st.error("Please upload a CSV file to make predictions.")
    else:
        try:
            X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                    'BMI', 'DiabetesPedigreeFunction', 'Age']]
            y = df['Outcome']
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            st.header("Personal Diabetes Prediction")
            pregnancies = st.number_input("Pregnancies", min_value=0)
            glucose = st.number_input("Glucose", min_value=0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0)
            skin_thickness = st.number_input("Skin Thickness", min_value=0)
            insulin = st.number_input("Insulin", min_value=0)
            bmi = st.number_input("BMI", min_value=0.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
            age = st.number_input("Age", min_value=0)
            
            if st.button("Predict"):
                prediction = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                              insulin, bmi, dpf, age]])
                if prediction == 1:
                    st.error("You have diabetes.")
                elif glucose > 150:
                    st.warning("You are at high risk of diabetes.")
                else:
                    st.success("You are at low risk of diabetes.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
