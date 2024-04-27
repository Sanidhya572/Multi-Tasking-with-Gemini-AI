from dotenv import load_dotenv
load_dotenv()  # Loading all the environment variables
from PIL import Image
import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model
model = genai.GenerativeModel("gemini-pro-vision")

def get_gimini_response(image):
    response = model.generate_content(image)
    return response.text

## Initialize Our Streamlit Application
st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

input_text = st.text_input("Input:", key="input")

uploaded_file = st.file_uploader("Choose a image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Tell me about the image")

# If submit is clicked 
if submit:
    response = get_gimini_response(image)
    st.subheader("The Response is: ")
    st.write(response)
