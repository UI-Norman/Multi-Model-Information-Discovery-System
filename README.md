# Multi-Model-Information-Discovery-System

A powerful, open-source document analysis and information retrieval system that combines local embeddings, vector search, and advanced LLM capabilities. Built with Streamlit, FAISS, and multiple API integrations for cost-effective document intelligence.

## Overview

The Information Discovery Assistant is designed to help users:
- **Upload and index** multiple document types (PDF, DOCX, TXT, PNG, JPG, JPEG)
- **Extract text** from documents including OCR for images
- **Perform semantic search** using local embeddings
- **Retrieve relevant information** with reranking for accuracy
- **Generate intelligent responses** using advanced LLMs
- **Maintain conversation context** with memory management
- **Store and recall** frequently asked questions

## Key Features

### Document Processing
- **Multi-format support**: PDF, DOCX, TXT, MD, PNG, JPG, JPEG
- **OCR capability**: Extract text from images using Tesseract
- **Intelligent chunking**: Configurable chunk size (default 300 words) with overlap
- **URL indexing**: Fetch and index content directly from web URLs
- **Error handling**: Graceful processing of corrupted or unreadable files

### Search & Retrieval
- **Local embeddings**: Uses `all-MiniLM-L6-v2` model (lightweight, free)
- **FAISS vector database**: Ultra-fast similarity search with IP metric
- **Cross-encoder reranking**: Improves result accuracy using `ms-marco-MiniLM-L-6-v2`
- **Dual-scoring system**: Combines vector similarity + rerank score 
- **Intelligent filtering**: Score threshold to eliminate irrelevant results

### AI & Language Models
- **Gemini 1.5 Flash**: Primary LLM with free tier support
- **Groq Llama 3.3 70B**: Fallback LLM for lower latency
- **Context-aware generation**: Includes conversation history and source attribution
- **Response consistency**: Maintains similar response length throughout conversations

### Knowledge Management
- **FAQ indexing**: Automatically builds FAQ database from queries and responses
- **Conversation history**: Maintains last 5 exchanges for context
- **Response caching**: Stores reranked results to avoid redundant processing
- **Metadata tracking**: Records source, upload date, chunk count, and document ID

## 📋 Prerequisites

### API Keys Required
- **Google Gemini API Key**: For LLM and OCR capabilities
- **Groq API Key**: For Llama model fallback

## 🚀 Installation

### 1. Clone or Download
```bash
git clone <repository-url>
cd info-discovery-assistant
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Windows:**
- Download installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Install and note the installation path

### 3. Set Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Access the application at `http://localhost:8501`

## Project Structure

```
info-discovery-assistant/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── README.md                        # This file
│
├── db/                             # Database directory (auto-created)
│   ├── documents.json              # Indexed document chunks (text)
│   ├── embeddings.pkl              # Vector embeddings (pickle format)
│   ├── metadata.json               # Metadata for each chunk (JSONL)
│   ├── faiss.index                 # FAISS vector index
│   └── responses.json              # Stored responses and queries
│
├── cache/                          # Cache directory (auto-created)
│   └── faq.json                    # FAQ index with embeddings
└── 
```

## How to Use

### Step 1: Index Documents

**Upload Files:**
1. Open sidebar panel on the left
2. Click "Drag and drop files here"
3. Select document(s) to upload (PDF, DOCX, TXT, images)
4. Click **Index** button
5. Wait for processing and success confirmation

### Step 2: Ask Questions

1. In the "Ask a question" section, enter your query
2. Click **Answer** button
3. System will:
   - Search indexed documents
   - Rerank results for relevance
   - Generate comprehensive response
   - Display sources and relevance scores

### Step 3: Review Conversation

- View chat history below the answer
- Click **View Sources** to see which documents contributed to the answer
- Check relevance scores and content previews
- Track FAQ matches vs. document-based responses

### Management Functions

| Action | Function | Effect |
|--------|----------|--------|
| **Index** | Index new documents | Adds to database |
| **Reindex (overwrite)** | Rebuild all embeddings | Refreshes FAISS index |
| **Clear Index** | Delete all data | Completely clears database |


## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                       │
│         (File upload, query input, results display)         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼─────────┐      ┌────────▼─────────┐
│ DataIngestion   │      │ ResponseGenerator│
│ Pipeline        │      │ + MemoryBuffer   │
│                 │      │                  │
│• Extract Text   │      │• Generate w/LLM  │
│• Chunk Text     │      │• Maintain History│
│• Index Docs     │      │• Multi-model     │
│• Embed Chunks   │      │  fallback        │
└────────┬────────┘      └──────────────────┘
         │
    ┌────┴──────────────────────────────┐
    │                                   │
┌───▼────────── ┐              ┌────────▼────────┐
│ VectorStore   │              │ QuestionIndex   │
│               │              │ (FAQ Manager)   │
│• Vector Search│              │                 │
│• Reranking    │              │• Search FAQ     │
│• Caching      │              │• Store Q&A      │
└────┬───────── ┘              └─────────────────┘
     │
┌────▼────────────────────────────────────────┐
│        FAISS Vector Index + Database        │
│  (documents.json, embeddings.pkl, faiss.db) │
└─────────────────────────────────────────────┘
```

### Data Flow

1. **Document Upload** → Extract Text → Chunk Text → Generate Embeddings
2. **Embedding Storage** → FAISS Index → Metadata Storage
3. **Query Input** → Encode Query → Vector Search → Rerank → LLM Generation
4. **Response** → Store in FAQ → Update Conversation History → Display


## Additional Resources

- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Groq API Docs](https://console.groq.com/docs)

## API Usage Notes

- **Gemini 2.5 Flash**: Free tier available 
- **Groq Llama 3.3 70B**: Free tier available with rate limits
- **Local embeddings**: Completely free, no API calls required
- **FAISS**: Open-source, no costs
