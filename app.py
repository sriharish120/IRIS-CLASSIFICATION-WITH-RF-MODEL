import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('IRIS.csv')
    return df

df = load_data()

# Prepare the data
X = df[['sepal_length', 'petal_length', 'petal_width']]
y = df['species']

# Train the model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Streamlit App
st.title('Iris Flower Species Prediction')

st.sidebar.header('Input Features')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()))
    petal_length = st.sidebar.slider('Petal Length', float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()))
    petal_width = st.sidebar.slider('Petal Width', float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()))
    
    data = {'sepal_length': sepal_length,
            'petal_length': petal_length,
            'petal_width': petal_width}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write(prediction[0])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Class Labels and their corresponding index number')
st.write(pd.DataFrame(df['species'].unique(), columns=['Species']))
