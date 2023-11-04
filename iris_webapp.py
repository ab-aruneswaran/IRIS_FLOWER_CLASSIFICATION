import streamlit as st
import pickle
import pandas as pd

lin_model=pickle.load(open('lin_model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))

def classify(num):
    if num<0.5:
        st.image('Iris_Setosa.jpg')
        return 'Iris-Setosa'
    elif num <1.5:
        st.image('Iris_Versicolor.jpg')
        return 'Iris-Versicolor'
    else:
        st.image('Iris_Virginica.jpg')
        return 'Iris-Virginica'
def main():
    st.title("WELCOME TO CLASSIFICATION WORLD")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">IRIS FLOWER CLASSIFICATION</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    sl=float(st.number_input('Enter Sepal Length'))
    sw=float(st.number_input('Enter Sepal Width'))
    pl=float(st.number_input('Enter Petal Length'))
    pw=float(st.number_input('Enter Petal Width'))
    inputs=[[sl,sw,pl,pw]]
    options = pd.DataFrame({
    'Model Name': ["Linear Regression", "Logistic Regression"],
    })
    ml_name = st.selectbox("Choose Your ML Model", options)
    if st.button('Classify'):
        if ml_name=='Linear Regression':
            st.success(classify(lin_model.predict(inputs)))
        else:
           st.success(classify(log_model.predict(inputs)))


if __name__=='__main__':
    main()