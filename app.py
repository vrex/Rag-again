import streamlit as st
import numpy as np
import faiss
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from newspaper import Article
import google.generativeai as genai

#CONFIGURATION
try:
    from secret_api_keys import GEMINI_API_KEY
    genai.configure(api_key=GEMINI_API_KEY)
except:
    st.error("API Key not found in secret_api_keys.py")

st.set_page_config(page_title="Gemini RAG App", layout="centered")

#DATA EXTRACTION

def extract_text(input_type, data):
    text = ""
    if input_type == "Link" and data:
        for url in data:
            if url:
                article = Article(url)
                article.download(); article.parse()
                text += f"\nSource: {url}\n{article.text}"
    elif input_type == "PDF" and data:
        reader = PdfReader(data)
        text = "".join([page.extract_text() for page in reader.pages])
    elif input_type == "DOCX" and data:
        doc = Document(data)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "TXT" and data:
        text = data.read().decode('utf-8')
    elif input_type == "Text":
        text = data
    return text

#CORE RAG ENGINE

def build_vector_store(text):
    # Split text into chunks
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    
    # Embed using Gemini (FIXED model name)
    res = genai.embed_content(model="models/gemini-embedding-001", content=chunks, task_type="retrieval_document")
    embeddings = np.array(res['embedding']).astype('float32')
    
    # Create FAISS Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# UI MAIN

def main():
    st.title("RAG Q&A App (Gemini Edition)")
    
    # 1. SELECTBOX UI (Just like your code)
    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
    input_data = None

    if input_type == "Link":
        num_links = st.number_input("Number of Links", min_value=1, max_value=5, step=1)
        input_data = [st.text_input(f"URL {i+1}") for i in range(num_links)]
    elif input_type == "Text":
        input_data = st.text_area("Enter your text")
    elif input_type == "PDF":
        input_data = st.file_uploader("Upload PDF", type=["pdf"])
    elif input_type == "TXT":
        input_data = st.file_uploader("Upload TXT", type=["txt"])
    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload DOCX", type=["docx"])

    if st.button("Proceed"):
        with st.spinner("Processing..."):
            raw_text = extract_text(input_type, input_data)
            if raw_text.strip():
                index, chunks = build_vector_store(raw_text)
                st.session_state["vectorstore"] = index
                st.session_state["chunks"] = chunks
                st.success("Knowledge base ready!")
            else:
                st.warning("No text found to process.")

    # 2. CHAT UI
    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            # Search
            q_emb = genai.embed_content(model="models/gemini-embedding-001", content=query, task_type="retrieval_query")['embedding']
            _, indices = st.session_state["vectorstore"].search(np.array([q_emb]).astype('float32'), k=3)
            context = " ".join([st.session_state["chunks"][i] for i in indices[0] if i != -1])
            
            # Answer with Gemini 3 Flash Preview
            model = genai.GenerativeModel('gemini-3-flash-preview')
            response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
            st.markdown(f"**Answer:**\n{response.text}")

if __name__ == "__main__":
    main()