from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(
    title="PDF Chat API",
    description=(
        "A FastAPI service that allows users to upload PDF documents and ask questions. "
        "The API processes the PDF, extracts text, and uses a conversational AI model to answer queries."
    ),
    version="1.0.0",
    contact={
        "name": "DPD",
        "url": "https://github.com/Prem-Dharshan/llm-hackathon",
        "email": "whizzkid.dpd@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)


app.mount("/static", StaticFiles(directory="static"), name="static")

embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

class UserQuestion(BaseModel):
    question: str

def get_pdf_text(pdf_docs: List[UploadFile]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks: List[str]):
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embed_model,
        persist_directory="chroma_db"
    )
    vector_store.persist()
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Ensure you provide all the details.
    If the answer is not in the provided context, respond with: "answer is not available in the context."
    Do not provide incorrect or fabricated answers.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = Ollama(
        model="mistral",
        temperature=0.3,
        max_tokens=1024,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the PDF Chat API!"}

@app.post("/ask_question/")
async def ask_question(
    user_question: str = Form(...),
    pdf_files: List[UploadFile] = File(...)
):
    try:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embed_model
        )

        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return {"answer": response["output_text"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing question: {str(e)}")
