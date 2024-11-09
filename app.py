import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
model = joblib.load("ingredient_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to classify each ingredient
def classify_individual_ingredients(ingredient_list):
    results = {}
    for ingredient in ingredient_list:
        input_tfidf = vectorizer.transform([ingredient])
        prediction = model.predict(input_tfidf)[0]
        results[ingredient] = prediction
    return results

# Streamlit App
st.title("Ingredient Health Status Classifier")
st.write("Enter a list of ingredients to classify them as Healthy, Moderate, or Unhealthy.")

# Text input for ingredients
input_text = st.text_area("Enter ingredients (comma-separated):", "")
if st.button("Classify"):
    ingredients = [ingredient.strip() for ingredient in input_text.split(",")]
    results = classify_individual_ingredients(ingredients)
    
    # Display the results
    for ingredient, status in results.items():
        st.write(f"{ingredient} - {status}")
