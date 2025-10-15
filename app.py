# Import necessary libraries for document processing, embeddings, OCR, and UI
import os
import json
import pickle
from datetime import datetime
from collections import deque
from pathlib import Path
import pdfplumber
import docx2txt
import pytesseract
from PIL import Image
import google.generativeai as genai
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
from dotenv import load_dotenv
import faiss
import requests
from bs4 import BeautifulSoup
import io
import tempfile
import uuid
import torch

# Load environment variables for API keys
load_dotenv()

# API Configuration
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "db"
CACHE_PATH = "cache"

# Initialize directories for database and cache
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

# Configure APIs and models
genai.configure(api_key=GEMINI_KEY)
groq_client = Groq(api_key=GROQ_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

# System prompt for LLM to ensure accurate and detailed responses
SYSTEM_PROMPT = """You are an advanced research and information discovery assistant with expertise in analyzing documents and providing comprehensive, accurate responses.

Your core capabilities and guidelines:

1. DOCUMENT ANALYSIS:
   - Thoroughly analyze all provided document contexts from PDF, DOCX, TXT, and image files
   - Extract and synthesize information from multiple sources
   - Identify key concepts, relationships, and patterns
   - Cross-reference information across documents for accuracy

2. RESPONSE QUALITY:
   - Provide detailed, well-structured answers that directly address the question
   - Include all relevant details from all document types, ensuring clarity and depth
   - Use clear, professional language appropriate for the subject matter
   - Break down complex concepts into understandable explanations
   - Organize information logically with proper flow and transitions

3. ACCURACY AND PRECISION:
   - Base all responses strictly on the provided document context
   - Clearly distinguish between facts from documents and logical inferences
   - If information is insufficient, acknowledge limitations explicitly
   - Avoid speculation or information not supported by the documents

4. RESPONSE STRUCTURE:
   - Start with a direct answer to the main question
   - Provide supporting details and elaboration from all relevant documents
   - Include specific examples or evidence from the documents
   - Conclude with synthesis or implications when appropriate
   - Use paragraphs for readability, avoid excessive bullet points

5. CONTEXTUAL AWARENESS:
   - Consider the broader context and intent of the question
   - Provide additional relevant information that enhances understanding
   - Connect related concepts across different parts of the documents
   - Anticipate follow-up questions and address them proactively

6. TECHNICAL CONTENT:
   - For technical queries, provide precise terminology and explanations
   - Include step-by-step explanations for processes or procedures
   - Define specialized terms when first introduced
   - Provide practical applications or implications

7. COMMUNICATION STYLE:
   - Maintain a professional, knowledgeable, and helpful tone
   - Be concise yet comprehensive, avoiding unnecessary verbosity
   - Use active voice and clear sentence structure
   - Adapt complexity level to the question's nature

8. LIMITATIONS:
   - Only answer based on provided document context
   - Do not make up information or cite sources not in the documents
   - If documents lack necessary information, state this clearly
   - Suggest what additional information would be needed for a complete answer

Your goal is to be the most helpful, accurate, and comprehensive assistant possible while maintaining strict adherence to the provided document context."""

# Data ingestion pipeline to handle document indexing and storage
class DataIngestionPipeline:
    def __init__(self):
        self.db_file = f"{DB_PATH}/documents.json"
        self.embeddings_file = f"{DB_PATH}/embeddings.pkl"
        self.metadata_file = f"{DB_PATH}/metadata.jsonl"
        self.faiss_index_file = f"{DB_PATH}/faiss.index"
        self.responses_file = f"{DB_PATH}/responses.json"
        self.load_db()
        self.pending_updates = []  # Buffer for lazy saving
    
    def load_db(self):
        # Load existing database files
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                self.documents = json.load(f)
        else:
            self.documents = []
        
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = []
        
        if os.path.exists(self.metadata_file):
            self.metadata = []
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        self.metadata.append(json.loads(line))
        else:
            self.metadata = []
        
        if os.path.exists(self.responses_file):
            with open(self.responses_file, 'r') as f:
                self.responses = json.load(f)
        else:
            self.responses = []
        
        if os.path.exists(self.faiss_index_file) and self.embeddings:
            self.faiss_index = faiss.read_index(self.faiss_index_file)
        else:
            self.faiss_index = faiss.IndexFlatIP(384)
    
    def extract_text(self, file):
        # Extract text from various file types with optimized processing
        try:
            if file.name.endswith('.pdf'):
                with pdfplumber.open(file) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                return text if text.strip() else "No extractable text in PDF"
            elif file.name.endswith('.docx'):
                text = docx2txt.process(file)
                return text if text.strip() else "No extractable text in DOCX"
            elif file.name.endswith(('.txt', '.md')):
                text = file.getvalue().decode('utf-8')
                return text if text.strip() else "No extractable text in TXT/MD"
            elif file.name.endswith(('.png', '.jpg', '.jpeg')):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                image = Image.open(tmp_path)
                text = pytesseract.image_to_string(image)
                os.unlink(tmp_path)
                return text if text.strip() else "No extractable text in image"
            else:
                text = file.getvalue().decode('utf-8', errors='ignore')
                return text if text.strip() else "No extractable text in file"
        except Exception as e:
            return f"Error extracting: {str(e)}"
    
    def chunk_text(self, text, chunk_size=300, overlap=50):
        # Split text into smaller chunks for faster processing
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def index_document(self, file):
        # Index document, create chunks, and store in database
        with st.spinner("Extracting text..."):
            text = self.extract_text(file)
        chunks = self.chunk_text(text)
        
        doc_id = f"doc_{uuid.uuid4()}"
        meta = {
            "id": doc_id,
            "filename": file.name,
            "upload_date": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "source": "upload"
        }
        
        with st.spinner("Generating embeddings..."):
            embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
            embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
        
        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append(chunk)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
            with open(self.metadata_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
                f.write('\n')  # Add newline for each metadata entry
            self.pending_updates.append(embedding)
        
        if self.pending_updates:
            embeddings_array = np.array(self.pending_updates).astype('float32')
            self.faiss_index.add(embeddings_array)
            self.pending_updates = []
        
        with st.spinner("Saving to database..."):
            self.save_db()
        return len(chunks), doc_id
    
    def index_url(self, url):
        # Index content from URL, create chunks, and store in database
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch URL: {response.status_code}")
        
        content_type = response.headers.get('Content-Type', '').lower()
        filename = url.split('/')[-1] or 'webpage'
        
        with st.spinner("Extracting text..."):
            if 'pdf' in content_type:
                file = io.BytesIO(response.content)
                file.name = filename if filename.endswith('.pdf') else f"{filename}.pdf"
                text = self.extract_text(file)
            elif 'html' in content_type or 'text/html' in content_type:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
            elif 'ms-word' in content_type or 'docx' in content_type:
                file = io.BytesIO(response.content)
                file.name = filename if filename.endswith('.docx') else f"{filename}.docx"
                text = self.extract_text(file)
            else:
                text = response.text
        
        chunks = self.chunk_text(text)
        
        doc_id = f"doc_{uuid.uuid4()}"
        meta = {
            "id": doc_id,
            "filename": url,
            "upload_date": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "source": "url"
        }
        
        with st.spinner("Generating embeddings..."):
            embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
            embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
        
        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append(chunk)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
            with open(self.metadata_file, 'a') as f:  # Append metadata to file
                json.dump(meta, f)
                f.write('\n')  # Add newline for each metadata entry
            self.pending_updates.append(embedding)
        
        if self.pending_updates:
            embeddings_array = np.array(self.pending_updates).astype('float32')
            self.faiss_index.add(embeddings_array)
            self.pending_updates = []
        
        with st.spinner("Saving to database..."):
            self.save_db()
        return len(chunks), doc_id
    
    def store_response(self, query, response, doc_id, sources):
        # Store query response as a new document in the database
        response_id = f"resp_{uuid.uuid4()}"
        meta = {
            "id": response_id,
            "filename": f"response_to_{query[:50]}.txt",
            "upload_date": datetime.now().isoformat(),
            "chunk_count": 1,
            "source": "response",
            "related_doc_id": doc_id,
            "query": query
        }
        
        embedding = embedding_model.encode(response)
        embedding = embedding / np.linalg.norm(embedding)
        self.documents.append(response)
        self.embeddings.append(embedding)
        self.metadata.append(meta)
        with open(self.metadata_file, 'a') as f:  # Append metadata to file
            json.dump(meta, f)
            f.write('\n')  # Add newline for each metadata entry
        self.pending_updates.append(embedding)
        
        self.responses.append({
            "id": response_id,
            "query": query,
            "response": response,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.pending_updates) >= 10:  # Lazy save after 10 updates
            embeddings_array = np.array(self.pending_updates).astype('float32')
            self.faiss_index.add(embeddings_array)
            self.pending_updates = []
            self.save_db()
    
    def save_db(self):
        # Save database files
        with open(self.db_file, 'w') as f:
            json.dump(self.documents, f)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.responses_file, 'w') as f:
            json.dump(self.responses, f)
        if self.faiss_index.ntotal > 0:
            faiss.write_index(self.faiss_index, self.faiss_index_file)
    
    def reindex(self):
        # Rebuild embeddings and FAISS index
        with st.spinner("Reindexing embeddings..."):
            self.embeddings = []
            for doc in self.documents:
                embedding = embedding_model.encode(doc)
                embedding = embedding / np.linalg.norm(embedding)
                self.embeddings.append(embedding)
        
        self.faiss_index = faiss.IndexFlatIP(384)
        if self.embeddings:
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.faiss_index.add(embeddings_array)
        
        with st.spinner("Saving to database..."):
            self.save_db()
    
    def clear_db(self):
        # Clear all database content
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.responses = []
        self.faiss_index = faiss.IndexFlatIP(384)
        self.pending_updates = []
        self.save_db()

# Vector store for similarity search using FAISS
class VectorStore:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.rerank_cache = {}  # Cache for reranking results
    
    def vector_search(self, query, top_k=10):
        # Perform vector search with reduced top_k
        if not self.pipeline.embeddings or self.pipeline.faiss_index.ntotal == 0:
            return []
        
        query_emb = embedding_model.encode(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        k = min(top_k, self.pipeline.faiss_index.ntotal)
        distances, indices = self.pipeline.faiss_index.search(query_emb, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score > 0.25:
                results.append({
                    "content": self.pipeline.documents[idx],
                    "metadata": self.pipeline.metadata[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        return results
    
    def rerank_results(self, query, results, top_k=8):
        # Rerank results with caching
        cache_key = (query, tuple(result['index'] for result in results))
        if cache_key in self.rerank_cache:
            return self.rerank_cache[cache_key]
        
        if not results:
            return []
        
        pairs = [[query, result['content']] for result in results]
        rerank_scores = reranker_model.predict(pairs)
        
        for i, result in enumerate(results):
            result['rerank_score'] = float(rerank_scores[i])
            result['combined_score'] = 0.4 * result['score'] + 0.6 * result['rerank_score']
        
        reranked = sorted(results, key=lambda x: x['combined_score'], reverse=True)[:top_k]
        self.rerank_cache[cache_key] = reranked
        if len(self.rerank_cache) > 100:  # Limit cache size
            self.rerank_cache.pop(next(iter(self.rerank_cache)))
        return reranked

# Question index for storing FAQs
class QuestionIndex:
    def __init__(self):
        self.index_file = f"{CACHE_PATH}/faq.json"
        self.load_faq()
    
    def load_faq(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                self.faq = json.load(f)
        else:
            self.faq = []
    
    def add_question(self, question, answer):
        self.faq.append({
            "id": len(self.faq),
            "question": question,
            "answer": answer,
            "embedding": embedding_model.encode(question).tolist()
        })
        self.save_faq()
    
    def save_faq(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.faq, f)
    
    def search_faq(self, query, threshold=0.70):
        query_emb = embedding_model.encode(query)
        matches = []
        for item in self.faq:
            sim = np.dot(query_emb, np.array(item['embedding']))
            if sim > threshold:
                matches.append((item, float(sim)))
        return sorted(matches, key=lambda x: x[1], reverse=True)

# Memory buffer for conversation history
class MemoryBuffer:
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, query, response):
        self.buffer.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context(self):
        return [{"role": "user" if i % 2 == 0 else "assistant", 
                "content": item["query"] if i % 2 == 0 else item["response"]}
                for i, item in enumerate(self.buffer)]

# Response generator using LLM
class ResponseGenerator:
    def __init__(self, use_gemini=True):
        self.use_gemini = use_gemini
        self.memory = MemoryBuffer()
        self.first_response_length = None
    
    def generate(self, query, retrieved_docs, use_gemini=None):
        if use_gemini is None:
            use_gemini = self.use_gemini
        
        # Truncate document content for faster LLM processing
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content'][:500] + ("..." if len(doc['content']) > 500 else "")
            context_parts.append(f"Document {i} (Relevance: {doc.get('combined_score', doc['score']):.2f}):\n{content}")
        context = "\n\n".join(context_parts)
        
        history_msgs = self.memory.get_context()
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history_msgs])
        
        prompt = f"""{SYSTEM_PROMPT}

Previous conversation:
{history_str}

Documents:
{context}

Question: {query}

Detailed Answer: Provide a comprehensive response that includes all relevant details from the documents, ensuring clarity and depth for all document types (PDF, DOCX, TXT, images)."""
        
        if self.first_response_length is not None:
            prompt += f"\nMake your response approximately {self.first_response_length} words long."
        
        try:
            if use_gemini:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                answer = response.text
            else:
                msg = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096
                )
                answer = msg.choices[0].message.content
            
            if self.first_response_length is None:
                self.first_response_length = len(answer.split())
            
            self.memory.add(query, answer)
            return answer
        
        except Exception as e:
            return self.generate(query, retrieved_docs, use_gemini=not use_gemini)

# Streamlit UI for the application
def main():
    st.set_page_config(page_title="Free-Tier Information Discovery Assistant", layout="wide")
    
    # Sidebar for indexing documents
    with st.sidebar:
        st.markdown("### Data Indexing Widget")
        st.markdown("*Upload documents (pdf, png, jpg, jpeg, txt, docx)*")
        st.markdown("*Click 'Index' to create chunks and store in DB*")
        
        uploaded_file = st.file_uploader(
            "Drag and drop files here",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'docx'],
            help="Limit 200MB per file"
        )
        url_input = st.text_input("Enter a website URL")

        
        col1, col2, col3 = st.columns(3)
        with col1:
            btn_index = st.button("Index", use_container_width=True)
        with col2:
            btn_reindex = st.button("Reindex (overwrite)", use_container_width=True)
        with col3:
            btn_clear = st.button("Clear Index", use_container_width=True)
    
    # Main content with UI styling
    st.markdown(
        "<h1 style='text-align: center; color: #87CEEB;'>Free-Tier Information Discovery Assistant</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #B0C4DE;'>Uses local embeddings (free) + FAISS vector database + Reranking + Gemini 1.5 Flash (free tier) for LLM/OCR.</p>",
        unsafe_allow_html=True
    )
    st.info("Embeddings run locally. FAISS for fast vector search. Cross-encoder reranking for improved accuracy. Supports centralized indexing from files and URLs.")
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataIngestionPipeline()
        st.session_state.vector_store = VectorStore(st.session_state.pipeline)
        st.session_state.faq = QuestionIndex()
        st.session_state.llm = ResponseGenerator(use_gemini=False)  # Prefer Grok for lower latency
        st.session_state.chat_history = []
        st.session_state.last_response = None
        st.session_state.last_reranked_docs = None
        st.session_state.current_doc_id = None
    
    pipeline = st.session_state.pipeline
    vector_store = st.session_state.vector_store
    faq = st.session_state.faq
    llm = st.session_state.llm
    
    # Handle indexing actions
    if btn_index:
        if uploaded_file:
            with st.spinner("Indexing file..."):
                chunk_count, doc_id = pipeline.index_document(uploaded_file)
                st.session_state.current_doc_id = doc_id
                st.success(f"Indexed {chunk_count} chunks from {uploaded_file.name}. Now you can ask questions!")
        elif url_input:
            with st.spinner("Indexing URL..."):
                try:
                    chunk_count, doc_id = pipeline.index_url(url_input)
                    st.session_state.current_doc_id = doc_id
                    st.success(f"Indexed {chunk_count} chunks from {url_input}. Now you can ask questions!")
                except Exception as e:
                    st.error(f"Error indexing URL: {str(e)}")
        else:
            st.warning("Please upload a file.")
    
    if btn_reindex:
        with st.spinner("Reindexing..."):
            pipeline.reindex()
            st.session_state.current_doc_id = None
            st.success("Reindexed all documents")
    
    if btn_clear:
        pipeline.clear_db()
        st.session_state.faq.faq = []
        st.session_state.chat_history = []
        st.session_state.current_doc_id = None
        st.success("Cleared database")
    
    # Chat interface
    st.markdown("### Chat with your documents")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation History")
        for i, chat in enumerate(st.session_state.chat_history[-5:]):
            with st.container():
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['response']}")
                if 'sources' in chat:
                    with st.expander("View Sources"):
                        for j, source in enumerate(chat['sources'][:3], 1):
                            st.write(f"Source {j}: {source['filename']}")
                            st.write(f"Relevance: {source['score']:.0%}")
                            st.write(f"Metadata: {source['metadata']}")
                            st.write(source['content'][:300] + "...")
                            st.divider()
                st.divider()
    
    # Query interface
    st.markdown("### Ask a question")
    
    col_query, col_btn = st.columns([4, 1])
    
    with col_query:
        user_query = st.text_input(
            "Enter your question:",
            placeholder="Type your question about the documents...",
            label_visibility="collapsed"
        )
    
    with col_btn:
        answer_btn = st.button("Answer", use_container_width=True, type="primary")
    
    # Process query when Answer button is clicked
    if answer_btn and user_query:
        if len(pipeline.documents) == 0:
            st.warning("Please index your documents first by uploading a file or URL and clicking 'Index'.")
        else:
            with st.spinner("Processing... Searching chunks..."):
                # Check FAQ first
                faq_results = faq.search_faq(user_query)
                
                if faq_results:
                    st.success("Found in FAQ")
                    response_text = faq_results[0][0]['answer']
                    
                    st.markdown("### Answer")
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 8px;
                        font-size: 16px;
                        line-height: 1.6;
                        max-height: 600px;
                        overflow-y: auto;
                    ">
                    {response_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"FAQ Match Confidence: {faq_results[0][1]:.0%}")
                    
                    st.session_state.chat_history.append({
                        'query': user_query,
                        'response': response_text,
                        'source': 'FAQ'
                    })
                else:
                    with st.spinner("Performing vector search..."):
                        docs = vector_store.vector_search(user_query)
                    
                    if docs:
                        with st.spinner("Reranking results..."):
                            reranked_docs = vector_store.rerank_results(user_query, docs)
                            st.session_state.last_reranked_docs = reranked_docs
                        
                        with st.spinner("Generating response..."):
                            response = llm.generate(user_query, reranked_docs)
                            st.session_state.last_response = response
                        
                        if st.session_state.current_doc_id:
                            sources = [{
                                'filename': doc['metadata']['filename'],
                                'score': doc['combined_score'],
                                'content': doc['content'],
                                'metadata': doc['metadata']
                            } for doc in reranked_docs[:3]]
                            pipeline.store_response(user_query, response, st.session_state.current_doc_id, sources)
                        
                        faq.add_question(user_query, response)
                        
                        st.markdown("### Answer")
                        st.markdown(f"""
                        <div style="
                            background-color: #f0f2f6;
                            padding: 20px;
                            border-radius: 8px;
                            font-size: 16px;
                            line-height: 1.6;
                            max-height: 600px;
                            overflow-y: auto;
                        ">
                        {response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        with st.expander("Source Documents (Reranked)"):
                            for i, doc in enumerate(reranked_docs[:5], 1):
                                st.write(f"**Source {i}:** {doc['metadata']['filename']}")
                                st.write(f"Vector Similarity: {doc['score']:.0%}")
                                st.write(f"Rerank Score: {doc['rerank_score']:.2f}")
                                st.write(f"Combined Score: {doc['combined_score']:.2f}")
                                st.write(f"Metadata: {doc['metadata']}")
                                st.write(f"Content Preview: {doc['content'][:300]}...")
                                st.divider()
                        
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'response': response,
                            'sources': [{
                                'filename': doc['metadata']['filename'],
                                'score': doc['combined_score'],
                                'content': doc['content'],
                                'metadata': doc['metadata']
                            } for doc in reranked_docs[:3]]
                        })
                    else:
                        st.warning("No relevant chunks found in indexed documents.")
    elif answer_btn and not user_query:
        st.warning("Please enter a question first.")
    
    # Document Statistics
    if pipeline.documents:
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", len(pipeline.documents))
        with col2:
            st.metric("FAISS Index Size", pipeline.faiss_index.ntotal)
        with col3:
            st.metric("FAQ Questions", len(faq.faq))
        with col4:
            st.metric("Conversation History", len(st.session_state.chat_history))

if __name__ == "__main__":
    main()