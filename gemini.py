import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import fitz  # Import fitz for PDF processing
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import faiss
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure GenerativeAI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro text generation model
text_model = genai.GenerativeModel("gemini-pro")

# Function to get text generation response
def get_text_response(question):
    response = text_model.generate_content(question)
    return response.text

# Function to load Gemini Pro vision model for image analysis
image_model = genai.GenerativeModel("gemini-pro-vision")

# Function to get image analysis response
def get_image_response(image):
    response = image_model.generate_content(image)
    return response.text

# Function for resume analysis
def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

# Function for extracting text content from uploaded PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Read the PDF file
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        # Initialize a list to hold the text of each page
        text_parts = []

        # Iterate over the pages of the PDF to extract the text
        for page in document:
            text_parts.append(page.get_text())

        # Concatenate the list into a single string with a space in between each part
        pdf_text_content = " ".join(text_parts)
        return pdf_text_content
    else:
        raise FileNotFoundError("No file uploaded")

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = faiss.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and display responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = faiss.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

# Streamlit application with improved theming and layout
st.set_page_config(
    page_title="Gemini Multi-Purpose App",
    page_icon=":robot:",
    layout="wide"
)

# Main content area
st.title("Welcome to Gemini Multi-Purpose App")

# Sidebar for selecting functionality
st.sidebar.title("Select Functionality")
selected_option = st.sidebar.selectbox(
    "Choose an option",
    ("Text Generation", "Image Analysis", "Resume Analysis", "Chat with PDF's")
)

if selected_option == "Text Generation":
    st.subheader("Gemini LLM Text Generation")
    input_text = st.text_input("Ask a question:")
    if st.button("Generate"):
        response = get_text_response(input_text)
        st.subheader("Response:")
        st.write(response)

elif selected_option == "Image Analysis":
    st.subheader("Gemini Image Analysis")
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze"):
            response = get_image_response(image)
            st.subheader("Analysis Result:")
            st.write(response)
    else:
        st.info("Please upload an image.")

elif selected_option == "Resume Analysis":
    st.subheader("Resume Analysis")
    # Resume analysis functionality
    st.header("Resume Analyzer")
    st.subheader('This Application helps you in your Resume Review with help of GEMINI AI [LLM]')
    input_text = st.text_input("Job Description: ", key="input")
    uploaded_file = st.file_uploader("Upload your Resume(PDF)...", type=["pdf"])
    pdf_content = ""

    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully")

    submit1 = st.button("Tell Me About the Resume")
    submit2 = st.button("How Can I Improvise my Skills")
    submit3 = st.button("What are the Keywords That are Missing")
    submit4 = st.button("Percentage match")
    input_prompt = st.text_input("Queries: Feel Free to Ask here")
    submit5 = st.button("Answer My Query")

    input_prompts = [
        """
         You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description.
         Please share your professional evaluation on whether the candidate's profile aligns with the role.
         Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        """,
        """
        You are a Technical Human Resource Manager with expertise in data science,
        your role is to scrutinize the resume in light of the job description provided.
        Share your insights on the candidate's suitability for the role from an HR perspective.
        Additionally, offer advice on enhancing the candidate's skills and identify areas where improvement is needed.
        """,
        """
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality,
        your task is to evaluate the resume against the provided job description. As a Human Resource manager,
        assess the compatibility of the resume with the role. Give me what are the keywords that are missing
        Also, provide recommendations for enhancing the candidate's skills and identify which areas require further development.
        """,
        """
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality,
        your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
        the job description. First the output should come as a percentage and then keywords missing and last final thoughts.
        """
    ]

    if submit1:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompts[0], pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload a PDF file to proceed.")

    elif submit2:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompts[1], pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload a PDF file to proceed.")

    elif submit3:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompts[2], pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload a PDF file to proceed.")

    elif submit4:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompts[3], pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload a PDF file to proceed.")

    elif submit5:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompt, pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload a PDF file to proceed.")

elif selected_option == "Chat with PDF's":
    st.subheader("Chat with PDF's")
    st.title("Chat with PDF using Gemini💬")
    st.write("Upload your PDF files and ask questions to get answers!")

    user_question = st.text_input("Ask a question")

    if user_question:
        user_input(user_question)

    st.sidebar.title("Menu")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if pdf_docs:
        if st.sidebar.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing completed!")
