# Use the official Python image with version 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies in the specified order
RUN pip install fastapi
RUN pip install python-dotenv
RUN pip install ollama  # Ollama's Python library for Mistral
RUN pip install langchain_community
RUN pip install PyPDF2
RUN pip install langchain-huggingface
RUN pip install uvicorn
RUN pip install streamlit
RUN pip install python-multipart
RUN pip install chromadb
RUN pip install sentence-transformers

# Copy the application code into the container
COPY . .

# Expose the application ports
EXPOSE 8000

# Set the environment variable for FastAPI
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI server
CMD ["uvicorn", "pdf_chat_api:app", "--host", "0.0.0.0", "--port", "8000"]
