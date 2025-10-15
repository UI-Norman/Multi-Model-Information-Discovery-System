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
from typing import List, Dict, Any, Tuple
import hashlib
import re
from difflib import SequenceMatcher

# Load environment variables for API keys
load_dotenv()

# API Configuration
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "db"
RESPONSES_PATH = "responses"

# Initialize directories
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(RESPONSES_PATH, exist_ok=True)

# Configure APIs
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
if GROQ_KEY:
    groq_client = Groq(api_key=GROQ_KEY)

# Initialize models with GPU support if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

# Enhanced System Prompt for comprehensive understanding
SYSTEM_PROMPT = """You are an advanced AI document assistant with exceptional comprehension abilities. Your mission is to provide thorough, insightful, and helpful responses.

CORE CAPABILITIES:
1. Understand ANY query style - formal, casual, broken English, keywords, phrases, or complete questions
2. Handle single documents or multiple documents seamlessly
3. Provide comprehensive, detailed, and well-structured responses
4. Extract deep insights and connections from documents
5. Answer questions even with minimal or vague information

RESPONSE PHILOSOPHY:
- ALWAYS provide value - never say "no information found" unless truly nothing exists
- Generate LONG, DETAILED responses (minimum 300 words for substantive queries)
- Use ALL available context to build comprehensive answers
- When information is limited, make intelligent inferences
- Structure responses clearly with sections, points, and explanations
- Connect information across multiple documents when relevant
- Provide examples and elaborations to enhance understanding

HANDLING DIFFERENT QUERY TYPES:
- Keywords only (e.g., "machine learning"): Explain the topic comprehensively using all related content
- Broken questions: Understand intent and provide full answers
- Summaries: Create detailed, multi-paragraph summaries
- Specific questions: Answer thoroughly with supporting details
- Vague queries: Interpret broadly and provide relevant information
- Document names: Focus ONLY on that document and ignore others

RESPONSE STRUCTURE FOR LONG ANSWERS:
1. Start with a clear, direct answer to the question
2. Provide detailed explanation with multiple paragraphs
3. Include specific information from the documents
4. Add context, background, and related information
5. Use examples where applicable
6. Conclude with synthesis or additional insights

Remember: Users prefer detailed, informative responses over brief summaries. Be thorough, be helpful, be comprehensive."""


class QueryAnalyzer:
    """Advanced query analyzer that understands any input style"""
    
    @staticmethod
    def analyze_query(query: str, all_doc_names: List[str] = None) -> Dict[str, Any]:
        """Analyze query with intelligent pattern recognition"""
        query_lower = query.lower().strip()
        
        analysis = {
            'original': query,
            'normalized': query_lower,
            'type': 'general',
            'keywords': [],
            'doc_name_mentioned': None,
            'intent': 'search',
            'query_length': len(query.split()),
            'is_keyword_only': False,
            'is_question': False
        }
        
        # IMPROVED: Better document name detection
        if all_doc_names:
            for doc_name in all_doc_names:
                doc_name_lower = doc_name.lower()
                doc_name_no_ext = os.path.splitext(doc_name_lower)[0]
                
                # Check for exact filename match or filename without extension
                if doc_name_lower in query_lower or doc_name_no_ext in query_lower:
                    analysis['doc_name_mentioned'] = doc_name
                    break
                
                # Check for partial matches (at least 60% similarity)
                words_in_query = query_lower.split()
                for word in words_in_query:
                    if len(word) > 3:  # Only check meaningful words
                        similarity = SequenceMatcher(None, word, doc_name_no_ext).ratio()
                        if similarity >= 0.6:
                            analysis['doc_name_mentioned'] = doc_name
                            break
                
                if analysis['doc_name_mentioned']:
                    break
        
        # Additional patterns to detect document references
        doc_reference_patterns = [
            r'(?:in|from|about|regarding|concerning)\s+(?:the\s+)?(?:document|doc|file|pdf|paper)\s+["\']?([^"\'?,.\n]+)["\']?',
            r'(?:document|doc|file|pdf|paper)\s+(?:called|named|titled)\s+["\']?([^"\'?,.\n]+)["\']?',
            r'["\']([^"\']+\.(?:pdf|docx|txt|md))["\']'
        ]
        
        for pattern in doc_reference_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                # Try to find matching document
                if all_doc_names:
                    for doc_name in all_doc_names:
                        if potential_name in doc_name.lower() or doc_name.lower() in potential_name:
                            analysis['doc_name_mentioned'] = doc_name
                            break
        
        # Detect if it's a question - more flexible patterns
        question_patterns = [
            r'^\s*(what|why|how|when|where|who|which|whose|whom)\b',
            r'\?$',
            r'\b(can|could|would|should|is|are|do|does|did|will|shall)\b.*\?',
            r'(tell me|explain|describe|show me|give me|find)',
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                analysis['is_question'] = True
                break
        
        # Detect if it's keyword-only (1-3 words, no question marks, no verbs)
        if analysis['query_length'] <= 3 and '?' not in query:
            analysis['is_keyword_only'] = True
        
        # Detect query type with expanded patterns
        summary_patterns = ['summary', 'summarize', 'summarise', 'overview', 'about', 'what is', 'tell me about', 'explain', 'describe']
        if any(pattern in query_lower for pattern in summary_patterns):
            analysis['type'] = 'summary'
            analysis['intent'] = 'summarize'
        elif any(word in query_lower for word in ['explain', 'define', 'meaning', 'what is', 'describe']):
            analysis['type'] = 'explanation'
            analysis['intent'] = 'explain'
        elif any(word in query_lower for word in ['find', 'search', 'look for', 'show me', 'give me']):
            analysis['type'] = 'search'
            analysis['intent'] = 'search'
        elif any(word in query_lower for word in ['list', 'all', 'show all', 'enumerate']):
            analysis['type'] = 'list'
            analysis['intent'] = 'list'
        
        # Extract keywords with minimal filtering
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'is', 'was', 'are', 'be', 'been', 'this', 'that', 'from', 'with'}
        
        words = re.findall(r'\b\w+\b', query_lower)
        analysis['keywords'] = [w for w in words if w not in stop_words and len(w) > 2]
        
        return analysis
    
    @staticmethod
    def expand_query(query: str, keywords: List[str], analysis: Dict) -> List[str]:
        """Generate intelligent query variations based on analysis"""
        variations = [query]
        
        # For keyword-only queries, create contextual variations
        if analysis['is_keyword_only']:
            if keywords:
                keyword_str = ' '.join(keywords)
                variations.extend([
                    f"What is {keyword_str}",
                    f"Explain {keyword_str}",
                    f"Tell me about {keyword_str}",
                    f"Information about {keyword_str}",
                    keyword_str
                ])
        
        # For questions, add simpler versions
        elif analysis['is_question']:
            if keywords:
                variations.append(' '.join(keywords))
        
        # For all queries, add keyword combination
        if keywords and len(keywords) > 1:
            variations.append(' '.join(keywords[:3]))  # Top 3 keywords
        
        return list(set(variations))  # Remove duplicates


class DocumentManager:
    """Enhanced document manager with better tracking"""
    
    def __init__(self):
        self.documents_file = f"{DB_PATH}/document_registry.json"
        self.load_registry()
    
    def load_registry(self):
        """Load document registry from disk"""
        if os.path.exists(self.documents_file):
            with open(self.documents_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def save_registry(self):
        """Save document registry to disk"""
        with open(self.documents_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def add_document(self, doc_id: str, filename: str, source: str, chunk_count: int, full_text: str = ""):
        """Add new document to registry"""
        self.registry[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "source": source,
            "chunk_count": chunk_count,
            "upload_date": datetime.now().isoformat(),
            "file_hash": self._generate_hash(filename),
            "preview": full_text[:500] if full_text else "",
            "word_count": len(full_text.split()) if full_text else 0
        }
        self.save_registry()
    
    def get_document(self, doc_id: str) -> Dict:
        """Get document metadata by ID"""
        return self.registry.get(doc_id, {})
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents in registry"""
        return list(self.registry.values())
    
    def get_all_document_names(self) -> List[str]:
        """Get all document filenames"""
        return [doc['filename'] for doc in self.registry.values()]
    
    def remove_document(self, doc_id: str):
        """Remove document from registry"""
        if doc_id in self.registry:
            del self.registry[doc_id]
            self.save_registry()
    
    def find_document_by_name(self, name: str, threshold: float = 0.5) -> List[str]:
        """Find documents by fuzzy name matching"""
        matches = []
        name_lower = name.lower().strip()
        
        for doc_id, doc_info in self.registry.items():
            filename = doc_info['filename'].lower()
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Exact substring match
            if name_lower in filename or name_lower in filename_no_ext:
                matches.append((doc_id, 1.0))
                continue
            
            # Fuzzy match on filename without extension
            similarity = SequenceMatcher(None, name_lower, filename_no_ext).ratio()
            if similarity >= threshold:
                matches.append((doc_id, similarity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in matches]
    
    def _generate_hash(self, filename: str) -> str:
        """Generate hash for filename"""
        return hashlib.md5(filename.encode()).hexdigest()


class ResponseLogger:
    """Logger for saving all responses to files"""
    
    def __init__(self):
        self.responses_dir = RESPONSES_PATH
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def save_response(self, query: str, response: str, metadata: Dict = None):
        """Save response to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.json"
        filepath = os.path.join(self.responses_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def get_all_responses(self) -> List[Dict]:
        """Get all saved responses"""
        responses = []
        for filename in sorted(os.listdir(self.responses_dir), reverse=True):
            if filename.endswith('.json'):
                filepath = os.path.join(self.responses_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    responses.append(json.load(f))
        return responses


class DataIngestionPipeline:
    """Enhanced pipeline for document processing"""
    
    def __init__(self):
        self.db_file = f"{DB_PATH}/documents.json"
        self.embeddings_file = f"{DB_PATH}/embeddings.pkl"
        self.metadata_file = f"{DB_PATH}/metadata.jsonl"
        self.faiss_index_file = f"{DB_PATH}/faiss.index"
        self.full_texts_file = f"{DB_PATH}/full_texts.json"
        self.doc_manager = DocumentManager()
        self.load_db()
        self.pending_updates = []
    
    def load_db(self):
        """Load database from disk"""
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
                    if line.strip():
                        self.metadata.append(json.loads(line))
        else:
            self.metadata = []
        
        if os.path.exists(self.full_texts_file):
            with open(self.full_texts_file, 'r') as f:
                self.full_texts = json.load(f)
        else:
            self.full_texts = {}
        
        if os.path.exists(self.faiss_index_file) and self.embeddings:
            self.faiss_index = faiss.read_index(self.faiss_index_file)
        else:
            self.faiss_index = faiss.IndexFlatIP(384)
    
    def extract_text(self, file) -> str:
        """Extract text from various file formats"""
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
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better context"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def index_document(self, file, progress_callback=None) -> Tuple[int, str]:
        """Index a document file"""
        if progress_callback:
            progress_callback("Extracting text...")
        
        text = self.extract_text(file)
        chunks = self.chunk_text(text)
        
        doc_id = f"doc_{uuid.uuid4()}"
        
        # Store full text
        self.full_texts[doc_id] = text
        
        if progress_callback:
            progress_callback(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
        embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            meta = {
                "id": doc_id,
                "filename": file.name,
                "upload_date": datetime.now().isoformat(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": "upload"
            }
            
            self.documents.append(chunk)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
            self.pending_updates.append(embedding)
        
        if self.pending_updates:
            embeddings_array = np.array(self.pending_updates).astype('float32')
            self.faiss_index.add(embeddings_array)
            self.pending_updates = []
        
        if progress_callback:
            progress_callback("Saving to database...")
        
        self.doc_manager.add_document(doc_id, file.name, "upload", len(chunks), text)
        self.save_db()
        
        return len(chunks), doc_id
    
    def index_url(self, url: str, progress_callback=None) -> Tuple[int, str]:
        """Index content from URL"""
        if progress_callback:
            progress_callback("Fetching URL...")
        
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch URL: {response.status_code}")
        
        content_type = response.headers.get('Content-Type', '').lower()
        filename = url.split('/')[-1] or 'webpage'
        
        if progress_callback:
            progress_callback("Extracting text...")
        
        if 'pdf' in content_type:
            file = io.BytesIO(response.content)
            file.name = filename if filename.endswith('.pdf') else f"{filename}.pdf"
            text = self.extract_text(file)
        elif 'html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
        else:
            text = response.text
        
        chunks = self.chunk_text(text)
        doc_id = f"doc_{uuid.uuid4()}"
        
        self.full_texts[doc_id] = text
        
        if progress_callback:
            progress_callback(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
        embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            meta = {
                "id": doc_id,
                "filename": url,
                "upload_date": datetime.now().isoformat(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": "url"
            }
            
            self.documents.append(chunk)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
            self.pending_updates.append(embedding)
        
        if self.pending_updates:
            embeddings_array = np.array(self.pending_updates).astype('float32')
            self.faiss_index.add(embeddings_array)
            self.pending_updates = []
        
        if progress_callback:
            progress_callback("Saving to database...")
        
        self.doc_manager.add_document(doc_id, url, "url", len(chunks), text)
        self.save_db()
        
        return len(chunks), doc_id
    
    def get_full_text(self, doc_id: str) -> str:
        """Get full text of a document"""
        return self.full_texts.get(doc_id, "")
    
    def get_all_full_texts(self) -> Dict[str, str]:
        """Get all full texts from all documents"""
        return self.full_texts.copy()
    
    def save_db(self):
        """Save database to disk"""
        with open(self.db_file, 'w') as f:
            json.dump(self.documents, f)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.full_texts_file, 'w') as f:
            json.dump(self.full_texts, f)
        if self.faiss_index.ntotal > 0:
            faiss.write_index(self.faiss_index, self.faiss_index_file)
        
        with open(self.metadata_file, 'w') as f:
            for meta in self.metadata:
                f.write(json.dumps(meta) + '\n')
    
    def clear_db(self):
        """Clear all data"""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.full_texts = {}
        self.faiss_index = faiss.IndexFlatIP(384)
        self.pending_updates = []
        self.doc_manager.registry = {}
        self.doc_manager.save_registry()
        self.save_db()


class VectorStore:
    """Enhanced vector store with intelligent multi-strategy search"""
    
    def __init__(self, pipeline: DataIngestionPipeline):
        self.pipeline = pipeline
        self.query_analyzer = QueryAnalyzer()
    
    def smart_search(self, query: str, top_k: int = 20, force_all_docs: bool = False) -> Tuple[List[Dict], Dict]:
        """
        IMPROVED: Better document-specific search with strict filtering
        force_all_docs: If True, searches ALL documents (for quick action buttons)
        """
        # Get all document names for better analysis
        all_doc_names = self.pipeline.doc_manager.get_all_document_names()
        
        # Analyze query with document names
        analysis = self.query_analyzer.analyze_query(query, all_doc_names)
        
        # CRITICAL: Check if specific document is mentioned
        doc_ids = None
        if analysis.get('doc_name_mentioned') and not force_all_docs:
            # Find the specific document
            doc_ids = self.pipeline.doc_manager.find_document_by_name(analysis['doc_name_mentioned'])
            if doc_ids:
                st.info(f"🎯 Searching only in document: **{analysis['doc_name_mentioned']}**")
        
        # Generate query variations
        query_variations = self.query_analyzer.expand_query(query, analysis['keywords'], analysis)
        
        # Perform multiple searches
        all_results = []
        
        # Strategy 1: Original query
        results1 = self.vector_search(query, top_k=top_k, threshold=0.05, doc_ids=doc_ids)
        all_results.extend(results1)
        
        # Strategy 2: Query variations (only if no specific document)
        if not doc_ids or force_all_docs:
            for variation in query_variations[1:3]:  # Limit to 2 variations
                results_var = self.vector_search(variation, top_k=top_k//2, threshold=0.05, doc_ids=doc_ids)
                all_results.extend(results_var)
        
        # Strategy 3: Keyword search
        if analysis['keywords']:
            for keyword in analysis['keywords'][:2]:  # Limit to 2 keywords
                results_kw = self.vector_search(keyword, top_k=top_k//3, threshold=0.0, doc_ids=doc_ids)
                all_results.extend(results_kw)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in all_results:
            if r['index'] not in seen:
                seen.add(r['index'])
                unique_results.append(r)
        
        # IMPROVED: Better fallback handling
        if not unique_results and self.pipeline.documents:
            if doc_ids:
                # If specific document requested but no results, get any chunks from that document
                for idx, meta in enumerate(self.pipeline.metadata):
                    if meta['id'] in doc_ids:
                        unique_results.append({
                            "content": self.pipeline.documents[idx],
                            "metadata": meta,
                            "score": 0.1,
                            "index": idx
                        })
                    if len(unique_results) >= 10:
                        break
            elif not force_all_docs:
                # Random sampling only if not force_all_docs
                import random
                num_samples = min(10, len(self.pipeline.documents))
                random_indices = random.sample(range(len(self.pipeline.documents)), num_samples)
                for idx in random_indices:
                    unique_results.append({
                        "content": self.pipeline.documents[idx],
                        "metadata": self.pipeline.metadata[idx],
                        "score": 0.1,
                        "index": int(idx)
                    })
        
        return unique_results[:top_k * 3], analysis
    
    def get_all_documents_content(self, max_per_doc: int = 5000) -> Tuple[List[Dict], Dict]:
        """Get content from ALL documents for comprehensive summaries"""
        all_docs = self.pipeline.doc_manager.get_all_documents()
        results = []
        
        for doc_info in all_docs:
            doc_id = doc_info['id']
            full_text = self.pipeline.get_full_text(doc_id)
            
            # Get representative chunks from this document
            doc_chunks = [
                (i, chunk, meta) 
                for i, (chunk, meta) in enumerate(zip(self.pipeline.documents, self.pipeline.metadata))
                if meta['id'] == doc_id
            ]
            
            # Add first few chunks
            for idx, chunk, meta in doc_chunks[:10]:
                results.append({
                    "content": chunk,
                    "metadata": meta,
                    "score": 1.0,
                    "index": idx,
                    "full_text_preview": full_text[:max_per_doc]
                })
        
        # Create analysis for "all documents" query
        analysis = {
            'original': 'ALL_DOCUMENTS_QUERY',
            'type': 'summary',
            'intent': 'summarize_all',
            'keywords': ['all', 'documents', 'summary'],
            'is_keyword_only': False,
            'is_question': False,
            'doc_name_mentioned': None  # Explicitly set to None for all-docs queries
        }
        
        return results, analysis
    
    def vector_search(self, query: str, top_k: int = 20, threshold: float = 0.05, doc_ids: List[str] = None) -> List[Dict]:
        """IMPROVED: Strict document filtering in vector search"""
        if not self.pipeline.embeddings or self.pipeline.faiss_index.ntotal == 0:
            return []
        
        query_emb = embedding_model.encode(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        # If doc_ids specified, search more chunks to ensure we get enough from target document
        search_k = min(top_k * 10 if doc_ids else top_k * 5, self.pipeline.faiss_index.ntotal)
        distances, indices = self.pipeline.faiss_index.search(query_emb, search_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score > threshold:
                meta = self.pipeline.metadata[idx]
                
                # STRICT FILTERING: If doc_ids specified, only include chunks from those documents
                if doc_ids and meta['id'] not in doc_ids:
                    continue
                
                results.append({
                    "content": self.pipeline.documents[idx],
                    "metadata": meta,
                    "score": float(score),
                    "index": int(idx)
                })
                
                # Stop once we have enough results
                if len(results) >= top_k:
                    break
        
        return results
    
    def rerank_results(self, query: str, results: List[Dict], top_k: int = 15) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not results:
            return []
        
        pairs = [[query, result['content']] for result in results]
        rerank_scores = reranker_model.predict(pairs)
        
        for i, result in enumerate(results):
            result['rerank_score'] = float(rerank_scores[i])
            # Balanced scoring
            result['combined_score'] = 0.3 * result['score'] + 0.7 * result['rerank_score']
        
        return sorted(results, key=lambda x: x['combined_score'], reverse=True)[:top_k]
    
    def get_comprehensive_context(self, results: List[Dict], query_analysis: Dict, max_length: int = 8000) -> str:
        """IMPROVED: Better context building with document-specific focus"""
        doc_groups = {}
        for result in results:
            doc_id = result['metadata']['id']
            if doc_id not in doc_groups:
                doc_info = self.pipeline.doc_manager.get_document(doc_id)
                doc_groups[doc_id] = {
                    'filename': result['metadata']['filename'],
                    'chunks': [],
                    'full_text': self.pipeline.get_full_text(doc_id)
                }
            doc_groups[doc_id]['chunks'].append(result)
        
        context_parts = []
        current_length = 0
        
        # CRITICAL: If specific document mentioned, ONLY include that document
        doc_name_mentioned = query_analysis.get('doc_name_mentioned')
        if doc_name_mentioned:
            # Find the document ID for the mentioned document
            target_doc_ids = self.pipeline.doc_manager.find_document_by_name(doc_name_mentioned)
            
            for doc_id in target_doc_ids:
                if doc_id in doc_groups:
                    data = doc_groups[doc_id]
                    header = f"\n{'='*80}\nDOCUMENT: {data['filename']}\n{'='*80}\n"
                    context_parts.append(header)
                    current_length += len(header)
                    
                    # Add substantial content from the specific document
                    full_text_portion = data['full_text'][:max_length - current_length - 500]
                    context_parts.append(f"\nFull Document Content:\n{full_text_portion}\n")
                    current_length += len(full_text_portion)
                    
                    # Add top relevant chunks
                    context_parts.append("\n\nMost Relevant Sections:\n")
                    for chunk in data['chunks'][:5]:
                        chunk_text = f"\n- {chunk['content']}\n"
                        if current_length + len(chunk_text) < max_length:
                            context_parts.append(chunk_text)
                            current_length += len(chunk_text)
                    
                    break  # ONLY process the first matching document
            
            # IMPORTANT: Return immediately, don't include other documents
            return "".join(context_parts)
        
        # For summary queries or all-document queries, include all documents
        if query_analysis.get('type') == 'summary' or query_analysis.get('intent') == 'summarize_all':
            for doc_id, data in doc_groups.items():
                header = f"\n{'='*80}\nDOCUMENT: {data['filename']}\n{'='*80}\n"
                context_parts.append(header)
                current_length += len(header)
                
                # Add substantial portion of full text
                full_text_portion = data['full_text'][:4000]
                context_parts.append(f"\nFull Document Content:\n{full_text_portion}\n")
                current_length += len(full_text_portion)
                
                if current_length >= max_length:
                    break
        else:
            # For regular queries, include chunks from all relevant documents
            for doc_id, data in doc_groups.items():
                header = f"\n{'='*80}\nDOCUMENT: {data['filename']}\n{'='*80}\n"
                context_parts.append(header)
                current_length += len(header)
                
                # Add relevant chunks
                for chunk in data['chunks'][:8]:
                    chunk_text = f"\n{chunk['content']}\n"
                    if current_length + len(chunk_text) < max_length:
                        context_parts.append(chunk_text)
                        current_length += len(chunk_text)
                
                if current_length >= max_length:
                    break
        
        return "".join(context_parts)


class ResponseGenerator:
    """Enhanced response generator for comprehensive answers"""
    
    def __init__(self, prefer_gemini: bool = True):
        self.prefer_gemini = prefer_gemini
        self.memory = deque(maxlen=5)
        self.response_logger = ResponseLogger()
    
    def generate(self, query: str, context: str, doc_count: int, query_analysis: Dict = None, action_type: str = None) -> str:
        """Generate comprehensive response using available context"""
        
        # Build conversation history
        history_str = "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {item}"
            for i, item in enumerate(self.memory)
        ])
        
        # IMPROVED: Better instruction based on document specificity
        doc_specific_instruction = ""
        if query_analysis and query_analysis.get('doc_name_mentioned'):
            doc_specific_instruction = f"""
**CRITICAL INSTRUCTION**: The user asked specifically about the document "{query_analysis['doc_name_mentioned']}".
You MUST answer using ONLY information from this specific document. DO NOT include information from other documents.
If you mention any information, it should be explicitly from "{query_analysis['doc_name_mentioned']}" only.
"""
        
        # Determine response strategy based on query analysis or action type
        if action_type:
            # Quick action buttons have specific instructions
            if action_type == "SUMMARIZE_ALL":
                instruction = """Provide a COMPREHENSIVE, DETAILED SUMMARY of ALL documents. Your summary MUST:

**Structure:**
- Start with an executive overview (2-3 paragraphs)
- Create a separate section for EACH document with its filename as a header
- Provide 3-5 paragraphs summarizing each document's content
- End with a conclusion that synthesizes information across all documents

**Content Requirements:**
- Be 600-800 words minimum
- Cover ALL major topics, themes, and key points from EVERY document
- Include specific details, statistics, examples, and important data mentioned
- Explain technical terms and provide context
- Show relationships and connections between documents where relevant

**Format:** Use clear paragraph structure with document headers."""

            elif action_type == "MAIN_TOPICS":
                instruction = """Identify and explain the MAIN TOPICS and THEMES from ALL documents. Your response MUST:

**Structure:**
1. **Overview of Topics** (1-2 paragraphs): Introduce the range of topics covered
2. **Topic-by-Topic Analysis:** For each major topic:
   - Topic name as a header
   - 2-3 paragraphs explaining the topic
   - Which documents discuss this topic
   - Key points and details about the topic
   - Examples or specific information mentioned
3. **Cross-Document Themes** (2-3 paragraphs): Topics that appear across multiple documents

**Requirements:**
- 500-700 words minimum
- Identify at least 5-8 major topics
- Provide detailed explanations, not just lists
- Connect topics to specific documents
- Include substantive content about each topic"""

            elif action_type == "KEY_POINTS":
                instruction = """Extract and explain ALL KEY POINTS from ALL documents. Your response MUST:

**Structure:**
- Start with an introduction (1 paragraph) explaining the scope
- **Document-by-Document Key Points:** For each document:
  - Document name as header
  - List 5-10 key points with detailed explanations
  - Each key point should be 2-3 sentences explaining its significance
  - Include context, examples, or supporting details
- **Synthesis** (2-3 paragraphs): Most important points across all documents

**Requirements:**
- 600-800 words minimum
- Extract at least 30-40 total key points across all documents
- Provide explanations, not just bullet lists
- Include specific information, data, or examples
- Organize clearly by document
- Show importance and relevance of each point"""

            elif action_type == "FULL_OVERVIEW":
                instruction = """Provide a COMPLETE, DETAILED OVERVIEW of EVERYTHING in ALL documents. This is the most comprehensive response. Your overview MUST:

**Structure:**
1. **Executive Summary** (3-4 paragraphs): High-level overview of all content
2. **Detailed Document Analysis:** For EACH document:
   - Document filename as major header
   - **Purpose & Context** (1-2 paragraphs)
   - **Main Content** (4-6 paragraphs covering all major sections/topics)
   - **Key Details** (2-3 paragraphs with specifics, data, examples)
   - **Significance** (1 paragraph)
3. **Comprehensive Synthesis** (3-4 paragraphs): 
   - Common themes across documents
   - Contrasts and unique perspectives
   - Overall insights and takeaways

**Requirements:**
- 800-1200 words minimum (this should be the LONGEST response)
- Leave nothing out - cover ALL significant content from ALL documents
- Include specific details, examples, statistics, methodologies
- Explain technical concepts thoroughly
- Show all relationships and connections
- Provide complete context and background information"""
            else:
                instruction = "Provide a comprehensive, detailed response."
        
        elif query_analysis:
            query_type = query_analysis.get('type', 'general')
            is_keyword_only = query_analysis.get('is_keyword_only', False)
            keywords = query_analysis.get('keywords', [])
            intent = query_analysis.get('intent', 'search')
            
            if query_type == 'summary' or intent == 'summarize_all':
                if query_analysis.get('doc_name_mentioned'):
                    instruction = f"""{doc_specific_instruction}

Provide a COMPREHENSIVE, DETAILED summary of the document "{query_analysis['doc_name_mentioned']}". Your summary should:
- Be at least 400-600 words long
- Cover ALL major topics and themes from this specific document
- Include specific details and key points
- Organize information into clear sections with headers
- Provide context and explanations for technical terms
- Include examples, statistics, or important data mentioned
- DO NOT mention or include information from other documents"""
                else:
                    instruction = """Provide a COMPREHENSIVE, DETAILED summary of ALL documents. Your summary should:
- Be at least 500-700 words long
- Cover ALL major topics and themes from EVERY document
- Include specific details and key points from each document
- List each document by name and summarize its content
- Organize information into clear sections with headers
- Provide context and explanations for technical terms
- Connect related information across documents
- Include examples, statistics, or important data mentioned"""
            
            elif is_keyword_only:
                keyword_str = ', '.join(keywords) if keywords else query
                if query_analysis.get('doc_name_mentioned'):
                    instruction = f"""{doc_specific_instruction}

The user searched for: "{keyword_str}" in document "{query_analysis['doc_name_mentioned']}"

Provide a COMPREHENSIVE, DETAILED explanation covering:
1. What this topic/concept is in the context of this document - 2-3 paragraphs
2. Key details and important information from this document - 3-4 paragraphs
3. Specific examples, applications, or use cases mentioned - 2-3 paragraphs
4. Related concepts within this document - 1-2 paragraphs
5. Any additional relevant context from this document - 1-2 paragraphs

Your response should be thorough (minimum 400 words), well-structured, and focus ONLY on this specific document."""
                else:
                    instruction = f"""The user searched for: "{keyword_str}"

Provide a COMPREHENSIVE, DETAILED explanation covering:
1. What this topic/concept is (definition and overview) - 2-3 paragraphs
2. Key details and important information from the documents - 3-4 paragraphs
3. Specific examples, applications, or use cases mentioned - 2-3 paragraphs
4. Related concepts and connections - 1-2 paragraphs
5. Any additional relevant context - 1-2 paragraphs

Your response should be thorough (minimum 400 words), well-structured, and informative."""
            
            else:
                if query_analysis.get('doc_name_mentioned'):
                    instruction = f"""{doc_specific_instruction}

Answer the user's question about document "{query_analysis['doc_name_mentioned']}" COMPREHENSIVELY and THOROUGHLY. Your response should:
- Be detailed and informative (aim for 300-500 words minimum)
- Directly address the question with a clear answer
- Provide supporting details and explanations FROM THIS DOCUMENT ONLY
- Include relevant examples and context from this specific document
- Structure your response with clear paragraphs
- Add insights and connections within this document
- Be helpful and complete - leave no question unanswered
- DO NOT include information from other documents"""
                else:
                    instruction = """Answer the user's question COMPREHENSIVELY and THOROUGHLY. Your response should:
- Be detailed and informative (aim for 300-500 words minimum)
- Directly address the question with a clear answer
- Provide supporting details and explanations
- Include relevant examples and context from the documents
- Structure your response with clear paragraphs
- Add insights and connections where appropriate
- Be helpful and complete - leave no question unanswered"""
        else:
            instruction = "Provide a comprehensive, detailed response."
        
        prompt = f"""{SYSTEM_PROMPT}

CONVERSATION HISTORY:
{history_str}

CURRENT TASK:
Analyzing {doc_count} document(s) to answer the user's query.

{instruction}

AVAILABLE CONTEXT FROM DOCUMENTS:
{context}

USER'S QUERY: {query}

INSTRUCTIONS:
1. Use ALL the context provided above - it's specifically retrieved for this query
2. Generate a LONG, DETAILED, COMPREHENSIVE response following the structure above
3. Be specific and include details from the documents
4. If a specific document was mentioned, answer ONLY from that document
5. If context is limited, make intelligent inferences and provide what you can
6. NEVER say "no information available" if there is ANY context above
7. Connect information across documents when relevant (unless specific document requested)
8. Provide examples and elaborations
9. For summary queries, make sure to cover ALL documents provided

RESPONSE:"""
        
        try:
            # Try primary API
            if self.prefer_gemini and GEMINI_KEY:
                answer = self._generate_gemini(prompt)
            elif GROQ_KEY:
                answer = self._generate_groq(prompt)
            else:
                answer = "Error: No API keys configured. Please set GEMINI_API_KEY or GROQ_API_KEY in your .env file."
        except Exception as e:
            # Fallback to alternative API
            try:
                if self.prefer_gemini and GROQ_KEY:
                    answer = self._generate_groq(prompt)
                elif GEMINI_KEY:
                    answer = self._generate_gemini(prompt)
                else:
                    answer = f"Error: Both APIs failed. Primary: {str(e)}"
            except Exception as e2:
                answer = f"Error: All APIs failed. Primary: {str(e)}, Fallback: {str(e2)}"
        
        # IMPROVED: Save response with better metadata
        metadata = {
            "doc_count": doc_count,
            "query_type": query_analysis.get('type') if query_analysis else action_type,
            "query_analysis": query_analysis,
            "action_type": action_type,
            "context_length": len(context),
            "response_length": len(answer),
            "doc_name_mentioned": query_analysis.get('doc_name_mentioned') if query_analysis else None,
            "specific_document_query": bool(query_analysis and query_analysis.get('doc_name_mentioned')) if query_analysis else False
        }
        self.response_logger.save_response(query, answer, metadata)
        
        # Update conversation memory
        self.memory.append(query)
        self.memory.append(answer)
        
        return answer
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'top_p': 0.95,
                'max_output_tokens': 8192,
            }
        )
        return response.text
    
    def _generate_groq(self, prompt: str) -> str:
        """Generate response using Groq API"""
        msg = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.7
        )
        return msg.choices[0].message.content


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Multi-Document RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #424242;
        font-size: 1.3em;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .doc-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .answer-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.8;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        max-height: 700px;
        overflow-y: auto;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #2196F3;
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-bubble {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">Multi-Document RAG System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">🔍 Ask anything - keywords, questions, summaries - in any style! Powered by APIs</div>',
        unsafe_allow_html=True
    )
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataIngestionPipeline()
        st.session_state.vector_store = VectorStore(st.session_state.pipeline)
        st.session_state.llm = ResponseGenerator(prefer_gemini=True)
        st.session_state.chat_history = []
    
    pipeline = st.session_state.pipeline
    vector_store = st.session_state.vector_store
    llm = st.session_state.llm
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'md', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, TXT, MD, or Image files"
        )
        
        # URL input
        url_input = st.text_input("📎 Or enter URL:", placeholder="https://example.com/document.pdf")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            btn_index = st.button("📥 Index All", use_container_width=True, type="primary")
        with col2:
            btn_clear = st.button("🗑️ Clear DB", use_container_width=True)
        
        st.divider()
        
        # Document list
        st.markdown("## 📚 Indexed Documents")
        all_docs = pipeline.doc_manager.get_all_documents()
        
        if all_docs:
            st.markdown(f"**Total: {len(all_docs)} document(s)**")
            for i, doc in enumerate(all_docs, 1):
                with st.expander(f"{i}. {doc['filename'][:40]}..."):
                    st.write(f"**ID:** `{doc['id'][:12]}...`")
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Words:** {doc.get('word_count', 'N/A')}")
                    st.write(f"**Source:** {doc['source']}")
                    st.write(f"**Date:** {doc['upload_date'][:10]}")
                    if doc.get('preview'):
                        st.text_area("Preview", doc['preview'][:200] + "...", height=100, key=f"preview_{i}")
        else:
            st.info("No documents indexed yet")
        
        st.divider()
        
        # Statistics
        st.markdown("## 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(all_docs))
        with col2:
            st.metric("Chunks", len(pipeline.documents))
        
        total_words = sum(doc.get('word_count', 0) for doc in all_docs)
        st.metric("Total Words", f"{total_words:,}")
        
        st.divider()
        
        # API Status
        st.markdown("## 🔌 API Status")
        if GEMINI_KEY:
            st.success("✅ Gemini API Configured")
        else:
            st.error("❌ Gemini API Not Configured")
        
        if GROQ_KEY:
            st.success("✅ Groq API Configured")
        else:
            st.error("❌ Groq API Not Configured")
        
        st.divider()
        
        # Query examples
        st.markdown("## 💡 Query Examples")
        st.markdown("""
        <div class="info-box">
        <b>Try these styles:</b><br><br>
        <b>Keywords:</b> "machine learning"<br>
        <b>Questions:</b> "What is AI?"<br>
        <b>Summaries:</b> "Summarize all"<br>
        <b>Specific doc:</b> "summary of report.pdf"<br>
        <b>Casual:</b> "tell me about doc1"<br>
        <b>Document name:</b> Just type the filename!
        </div>
        """, unsafe_allow_html=True)
    
    # Handle indexing
    if btn_index:
        if uploaded_files or url_input:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_chunks = 0
            total_processed = 0
            
            # Process uploaded files
            if uploaded_files:
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"⏳ Processing {file.name}...")
                    
                    def progress_callback(msg):
                        status_text.text(f"⏳ {file.name}: {msg}")
                    
                    try:
                        chunk_count, doc_id = pipeline.index_document(file, progress_callback)
                        total_chunks += chunk_count
                        total_processed += 1
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    except Exception as e:
                        st.error(f"❌ Error indexing {file.name}: {str(e)}")
            
            # Process URL
            if url_input:
                status_text.text(f"⏳ Processing URL...")
                
                def progress_callback(msg):
                    status_text.text(f"⏳ URL: {msg}")
                
                try:
                    chunk_count, doc_id = pipeline.index_url(url_input, progress_callback)
                    total_chunks += chunk_count
                    total_processed += 1
                    progress_bar.progress(1.0)
                except Exception as e:
                    st.error(f"❌ Error indexing URL: {str(e)}")
            
            status_text.empty()
            progress_bar.empty()
            
            if total_processed > 0:
                st.markdown(f"""
                <div class="success-box">
                <b>✅ Successfully indexed {total_processed} document(s)</b><br>
                Total chunks created: {total_chunks}<br>
                Saved to database
                </div>
                """, unsafe_allow_html=True)
                st.rerun()
        else:
            st.warning("⚠️ Please upload files or enter a URL first")
    
    if btn_clear:
        pipeline.clear_db()
        st.session_state.chat_history = []
        st.success("✅ Database cleared successfully!")
        st.rerun()
    
    # Main content area
    st.markdown("## 💬 Chat with Your Documents")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Recent Conversations")
        for chat in st.session_state.chat_history[-5:]:
            # Question
            st.markdown(f"""
            <div class="chat-bubble">
            <b>❓ Question:</b> {chat['query']}
            </div>
            """, unsafe_allow_html=True)
            
            # Answer
            st.markdown(f'<div class="answer-box">{chat["response"]}</div>', unsafe_allow_html=True)
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📄 {chat.get('doc_count', 0)} document(s)")
            with col2:
                st.caption(f"📝 {chat.get('chunk_count', 0)} chunks used")
            with col3:
                st.caption(f"✍️ {chat.get('response_length', 0)} characters")
            
            if 'query_type' in chat:
                st.caption(f"🏷️ Query type: **{chat['query_type']}**")
            
            st.divider()
    
    # Query input section
    st.markdown("### Ask Your Question")
    
    col_query, col_btn = st.columns([5, 1])
    
    with col_query:
        user_query = st.text_input(
            "Type anything:",
            placeholder="Type keywords, questions, document names, or requests in any style...",
            label_visibility="collapsed",
            key="query_input"
        )
    
    with col_btn:
        ask_btn = st.button("🔍 Ask", use_container_width=True, type="primary")
    
    # Quick action buttons - EACH WITH UNIQUE BEHAVIOR
    st.markdown("**⚡ Quick Actions (Work on ALL Documents):**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_action_query = None
    action_type = None
    
    with col1:
        if st.button("📋 Summarize All", use_container_width=True, help="Comprehensive summary of all documents"):
            quick_action_query = "Give me a comprehensive detailed summary of ALL documents"
            action_type = "SUMMARIZE_ALL"
    
    with col2:
        if st.button("🎯 Main Topics", use_container_width=True, help="Identify main topics and themes"):
            quick_action_query = "What are the main topics and themes covered in ALL documents"
            action_type = "MAIN_TOPICS"
    
    with col3:
        if st.button("🔑 Key Points", use_container_width=True, help="Extract all key points"):
            quick_action_query = "Extract and explain all key points from ALL documents"
            action_type = "KEY_POINTS"
    
    with col4:
        if st.button("📖 Full Overview", use_container_width=True, help="Complete detailed overview of everything"):
            quick_action_query = "Give me a complete detailed overview of everything in ALL documents"
            action_type = "FULL_OVERVIEW"
    
    # Process quick action or regular query
    if quick_action_query:
        if len(pipeline.documents) == 0:
            st.warning("⚠️ Please index some documents first!")
        else:
            with st.spinner("🔄 Gathering content from ALL documents..."):
                # Get ALL documents content
                results, query_analysis = vector_store.get_all_documents_content()
            
            if results:
                with st.spinner("📊 Processing content from all documents..."):
                    # Get document information
                    doc_ids = set([r['metadata']['id'] for r in results])
                    doc_count = len(doc_ids)
                    
                    st.success(f"✅ Processing content from ALL {doc_count} document(s)")
                
                with st.spinner(f"🤖 Generating {action_type.replace('_', ' ').title()} response..."):
                    # Build context from ALL documents
                    context = vector_store.get_comprehensive_context(results, query_analysis, max_length=8000)
                    
                    # Generate response WITH SPECIFIC ACTION TYPE
                    response = llm.generate(quick_action_query, context, doc_count, query_analysis, action_type=action_type)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
                
                # Metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Documents", doc_count)
                with col2:
                    st.metric("📝 Chunks Used", len(results))
                with col3:
                    st.metric("✍️ Response Length", len(response))
                with col4:
                    st.metric("💾 Saved to File", "✓")
                
                # Source documents
                with st.expander("📚 All Documents Analyzed", expanded=False):
                    for doc_id in doc_ids:
                        doc_info = pipeline.doc_manager.get_document(doc_id)
                        
                        st.markdown(f'<div class="doc-card">', unsafe_allow_html=True)
                        st.markdown(f"### 📄 {doc_info.get('filename', 'Unknown')}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Total Chunks:** {doc_info.get('chunk_count', 'N/A')}")
                        with col2:
                            st.write(f"**Word Count:** {doc_info.get('word_count', 'N/A')}")
                        with col3:
                            st.write(f"**Source:** {doc_info.get('source', 'N/A')}")
                        
                        if doc_info.get('preview'):
                            st.text_area(
                                "Preview",
                                doc_info['preview'][:400] + "...",
                                height=150,
                                key=f"all_docs_preview_{doc_id}",
                                label_visibility="collapsed"
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Save to history
                st.session_state.chat_history.append({
                    'query': quick_action_query,
                    'response': response,
                    'doc_count': doc_count,
                    'chunk_count': len(results),
                    'query_type': action_type,
                    'response_length': len(response)
                })
    
    elif ask_btn and user_query:
        if len(pipeline.documents) == 0:
            st.warning("⚠️ Please index some documents first!")
        else:
            with st.spinner("🔍 Analyzing your query..."):
                # Smart search - DO NOT force all docs for user queries
                results, query_analysis = vector_store.smart_search(user_query, top_k=20, force_all_docs=False)
            
            if results:
                with st.spinner("📊 Ranking and processing results..."):
                    # Rerank results
                    reranked = vector_store.rerank_results(user_query, results, top_k=15)
                    
                    # Get document information
                    doc_ids = set([r['metadata']['id'] for r in reranked])
                    doc_count = len(doc_ids)
                    
                    # Display info about what was found
                    if query_analysis.get('doc_name_mentioned'):
                        st.success(f"✅ Found {len(reranked)} relevant chunks from document: **{query_analysis['doc_name_mentioned']}**")
                    else:
                        st.success(f"✅ Found {len(reranked)} relevant chunks")
                
                with st.spinner("🤖 Generating comprehensive answer..."):
                    # Build context
                    context = vector_store.get_comprehensive_context(reranked, query_analysis, max_length=6000)
                    
                    # Generate response
                    response = llm.generate(user_query, context, doc_count, query_analysis)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
                
                # Metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Documents", doc_count)
                with col2:
                    st.metric("📝 Chunks Used", len(reranked))
                with col3:
                    st.metric("✍️ Response Length", len(response))
                with col4:
                    st.metric("💾 Saved to File", "✓")
                
                # Additional info if specific document was queried
                if query_analysis.get('doc_name_mentioned'):
                    st.info(f"ℹ️ This answer is based exclusively on: **{query_analysis['doc_name_mentioned']}**")
                
                # Source documents
                with st.expander("📚 Source Documents & Chunks", expanded=False):
                    for doc_id in doc_ids:
                        doc_info = pipeline.doc_manager.get_document(doc_id)
                        doc_chunks = [r for r in reranked if r['metadata']['id'] == doc_id]
                        
                        st.markdown(f'<div class="doc-card">', unsafe_allow_html=True)
                        st.markdown(f"### 📄 {doc_info.get('filename', 'Unknown')}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Chunks Used:** {len(doc_chunks)}")
                        with col2:
                            avg_score = np.mean([c['combined_score'] for c in doc_chunks]) if doc_chunks else 0
                            st.write(f"**Avg Score:** {avg_score:.3f}")
                        with col3:
                            st.write(f"**Total Chunks:** {doc_info.get('chunk_count', 'N/A')}")
                        
                        st.markdown("**Top Relevant Chunks:**")
                        for i, chunk in enumerate(doc_chunks[:5], 1):
                            with st.container():
                                st.markdown(f"**Chunk {i}** - Relevance: `{chunk['combined_score']:.3f}`")
                                st.text_area(
                                    f"Content",
                                    chunk['content'][:600] + ("..." if len(chunk['content']) > 600 else ""),
                                    height=120,
                                    key=f"chunk_{doc_id}_{i}_{hash(chunk['content'][:20])}",
                                    label_visibility="collapsed"
                                )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Relevance scores
                with st.expander("📊 Relevance Scores Analysis", expanded=False):
                    st.markdown("**Top 10 Most Relevant Chunks:**")
                    for i, result in enumerate(reranked[:10], 1):
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.write(f"{i}. {result['metadata']['filename'][:40]}...")
                        with col2:
                            st.write(f"🎯 {result['combined_score']:.3f}")
                        with col3:
                            st.write(f"📐 {result['score']:.3f}")
                        with col4:
                            st.write(f"🔄 {result['rerank_score']:.3f}")
                
                # Save to history
                st.session_state.chat_history.append({
                    'query': user_query,
                    'response': response,
                    'doc_count': doc_count,
                    'chunk_count': len(reranked),
                    'query_type': query_analysis.get('type', 'general'),
                    'response_length': len(response)
                })
            
            else:
                # No results found
                if query_analysis.get('doc_name_mentioned'):
                    st.warning(f"⚠️ No relevant information found in document: **{query_analysis['doc_name_mentioned']}**")
                    st.markdown("**Possible reasons:**")
                    st.markdown("""
                    - The document name might be spelled differently
                    - The document might not contain information about this topic
                    - Try checking the exact filename in the sidebar
                    """)
                else:
                    st.warning("⚠️ No relevant information found. Try:")
                    st.markdown("""
                    - Using different keywords or phrases
                    - Asking a more general question
                    - Using the 'Summarize All' button to see document contents
                    - Checking if documents were indexed correctly
                    """)
                st.info(f"ℹ️ Database contains: {len(all_docs)} documents with {len(pipeline.documents)} chunks")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #757575; padding: 20px;">
    <b> Powered by Advanced AI APIs</b><br>
    Gemini 2.0 Flash | Groq LLama 3.3 70B | All responses saved to <code>responses/</code> folder<br>
    <small>Enhanced Query Understanding | Multi-Strategy Search | Intelligent Response Generation</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()