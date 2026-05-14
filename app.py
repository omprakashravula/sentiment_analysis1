import streamlit as st
import pickle

# Load your saved BernoulliNB model and vectorizer
model = pickle.load(open('bernoulli_model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Sentiment Analysis App")
user_input = st.text_input("Enter a review:")

if st.button("Predict"):
    # Process and Predict (as requested in Task 7)
    data = cv.transform([user_input.lower()]).toarray()
    prediction = model.predict(data)
    result = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"Result: **{result}**")