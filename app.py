import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Validate API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Error configuring Google API: {str(e)}")
    raise
# Set model name with fallback
model_name = os.getenv("MODEL_NAME", "gemini-pro")
# Initialize BGE embeddings
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        raise
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        raise
def get_vector_store(text_chunks):
    try:
        vector_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=embed_model,
            persist_directory="chroma_db"
        )
        vector_store.persist()
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise
def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        raise
def user_input(user_question):
    try:
        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embed_model
        )
        
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents":docs, "question": user_question},
            return_only_outputs=True
        )
        
        print(response)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        raise
def main():
    try:
        st.set_page_config("Chat PDF")
        st.header("Chat with PDF using Gemini💁")
        # Display API key status
        if GOOGLE_API_KEY:
            st.sidebar.success("API key loaded successfully")
        else:
            st.sidebar.error("API key not found")
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question)
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file")
                    return
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
main()