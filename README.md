# AI ML Olympiad Hackathon

## PDF LLM 

A FastAPI service that allows users to upload PDF documents and ask questions about their content. The API uses conversational AI to answer queries based on the uploaded PDFs.

## Features
- Upload PDF documents and extract content.
- Query the extracted content using conversational AI.
- Serve the application standalone or via Docker.

---

## Requirements
- Python 3.10+
---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/pdf-chat-api.git
cd pdf-chat-api
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables
Create a `.env` file in the project root and add your **Google API Key**:
```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Run the Applications

### Run FastAPI Server (Standalone)
1. Run the FastAPI server:
    ```bash
    uvicorn pdf_chat_api:app --host 0.0.0.0 --port 8000
    ```
2. Open your browser and navigate to:
    - API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
    - Root Endpoint: [http://localhost:8000/](http://localhost:8000/)

---

## Docker Setup

### 1. Build the Docker Image
```bash
docker build -t pdf-chat-api .
```

### 2. Run the Docker Container
```bash
docker run -d --name pdf-chat-api -p 8000:8000 pdf-chat-api
```

### 3. Access the Application
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Root Endpoint: [http://localhost:8000/](http://localhost:8000/)

---

## Notes

### Stop and Remove Docker Container
```bash
docker stop pdf-chat-api
docker rm pdf-chat-api
```

### Optional: View Logs
```bash
docker logs pdf-chat-api
```

---

## Streamlit App (Optional)
This repository doesn't currently include a Streamlit app, but here's how you can run it if added:
1. Save your Streamlit app as `app.py`.
2. Run the Streamlit server:
    ```bash
    streamlit run app.py
    ```
3. Open your browser to [http://localhost:8501/](http://localhost:8501/).

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
