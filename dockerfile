# Use the official Python image with version 3.11.9 as the base image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "gemini.py"]
