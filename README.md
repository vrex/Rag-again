RAG Q&A App (Using Gemini Edition)

A memory-efficient Retrieval-Augmented Generation (RAG) application built with Streamlit and Google Gemini. This app allows you to upload documents or provide links, then ask questions and get AI-powered answers based on your content.

 Features

- **Multiple Input Formats**: Support for PDF, DOCX, TXT files, web links, and direct text input
- **Efficient Vector Search**: Uses FAISS for fast similarity search
- **Gemini Integration**: Powered by Google Gemini 3 Flash Preview for answer generation
- **Memory-Friendly**: Optimized for systems with limited RAM (8GB+)
- **Easy-to-Use Interface**: Clean Streamlit web interface

Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

Installation
 1. Clone the repository

```bash
git clone <your-repo-url>
cd Rag-again
```
 2. Create a virtual environment

```bash
python -m venv rag_env
```
 3. Activate the virtual environment

**Windows:**
```bash
rag_env\Scripts\activate
```

**Linux/Mac:**
```bash
source rag_env/bin/activate
```

 4. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

 5. Set up API key

Create a file named `secret_api_keys.py` in the project root:

```python
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Option 1: Using the batch file (Windows)

Simply double-click `run.bat` or run:

```bash
run.bat
```

 Option 2: Manual command

```bash
# Activate virtual environment
rag_env\Scripts\activate  # Windows
# or
source rag_env/bin/activate  # Linux/Mac

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

 How to Use

1. **Select Input Type**: Choose from Link, PDF, Text, DOCX, or TXT
2. **Provide Content**: 
   - Upload a file, or
   - Enter URLs (for Link), or
   - Paste text directly
3. **Click "Proceed"**: The app will process and index your content
4. **Ask Questions**: Enter your question and click "Submit" to get AI-powered answers

 Architecture


Components

- **Embeddings**: Google Gemini Embedding API (`models/gemini-embedding-001`)
  - Converts text chunks into vector representations
  - Used for semantic search in the document corpus

- **Vector Store**: FAISS (Facebook AI Similarity Search)
  - Fast, efficient similarity search
  - Runs locally on CPU

- **Answer Generation**: Google Gemini 3 Flash Preview
  - Generates answers based on retrieved context
  - Remote API call (requires internet)


Workflow

1. **Document Processing**:
   - Text extraction → Chunking (1000 char chunks) → Embedding → FAISS index

2. **Question Answering**:
   - Question embedding → Similarity search → Context retrieval → Answer generation

 Dependencies

- `streamlit` - Web interface
- `google-generativeai` - Gemini API integration
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF processing
- `python-docx` - DOCX file handling
- `newspaper3k` - Web article extraction
- `numpy` - Numerical operations

See `requirements.txt` for specific versions.

Configuration

Model Settings

- **Embedding Model**: `models/gemini-embedding-001`
- **Generation Model**: `gemini-3-flash-preview`
- **Chunk Size**: 1000 characters
- **Retrieval**: Top 3 most relevant chunks
 Memory Usage

- **Indexing**: Low-moderate CPU usage
- **Idle**: Very low memory footprint
- **Answer Generation**: Network-bound (API call)
- **Total RAM**: ~1-1.5 GB typical usage

 Troubleshooting

 API Key Error
- Ensure `secret_api_keys.py` exists with your `GEMINI_API_KEY`
- Verify the API key is valid and has quota remaining

Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

 Memory Issues
- Reduce chunk size in `build_vector_store()` function
- Process smaller documents
- Close other applications

 License

This project is open source and available under the MIT License.





