from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.staticfiles import StaticFiles

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

app = FastAPI(
    title="PDF Chat API",
    description=(
        "A FastAPI service that allows users to upload PDF documents and ask questions. "
        "The API processes the PDF, extracts text, and uses a conversational AI model to answer queries."
    ),
    version="1.0.0",
    contact={
        "name": "Your Name",
        "url": "https://yourwebsite.com",
        "email": "your.email@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Serve static files (including favicon)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Google API client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise Exception(f"Error configuring Google API: {str(e)}")

# Initialize BGE embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Define a model for the user input question
class UserQuestion(BaseModel):
    question: str

# Function to extract text from PDF
def get_pdf_text(pdf_docs: List[UploadFile]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create vector store
def get_vector_store(text_chunks: List[str]):
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embed_model,
        persist_directory="chroma_db"
    )
    vector_store.persist()
    return vector_store

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",  # Or your custom model
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Root route
@app.get("/")
async def read_root():
    """
    Welcome route for the PDF Chat API.
    """
    return {"message": "Welcome to the PDF Chat API!"}

# Endpoint to upload PDF and ask questions
@app.post("/ask_question/")
async def ask_question(
    user_question: str = Form(...),
    pdf_files: List[UploadFile] = File(...)
):
    """
    Upload a PDF document and ask a question about its content.
    - **user_question**: The question to ask based on the PDF content.
    - **pdf_files**: One or more PDF files to process.
    """
    try:
        # Extract text from PDF files
        raw_text = get_pdf_text(pdf_files)
        
        # Split the extracted text into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create vector store from text chunks
        get_vector_store(text_chunks)

        # Load vector store for similarity search
        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embed_model
        )

        # Perform similarity search based on the user's question
        docs = vector_store.similarity_search(user_question)
        
        # Get the conversational chain for answering questions
        chain = get_conversational_chain()

        # Get the answer from the chain
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        return {"answer": response["output_text"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing question: {str(e)}")
