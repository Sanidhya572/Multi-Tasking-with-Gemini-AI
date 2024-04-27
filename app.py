from dotenv import load_dotenv
load_dotenv()  # Loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model
model = genai.GenerativeModel("gemini-pro")

def get_gimini_response(question):
    response = model.generate_content(question)
    return response.text

## Initialize Our Streamlit Application
st.set_page_config(page_title="Q and A Demo")
st.header("Gemini LLM Application")

input_text = st.text_input("Input:", key="input")
submit = st.button("Ask the question")

## When submit is clicked
if submit:
    response = get_gimini_response(input_text)
    st.subheader("The Response is")
    st.write(response)
