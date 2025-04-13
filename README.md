# Multimodal RAG System: Document Understanding with Text and Image Analysis

## Overview

This Multimodal Retrieval-Augmented Generation (RAG) system is a sophisticated solution for intelligent document processing and querying, capable of handling both text and image content from PDFs using advanced AI technologies.

## Key Features

- 🔍 **Multimodal Ingestion**: 
  - Extract text and images from PDF documents
  - Robust preprocessing and cleaning of extracted content
  - Chunking of text for semantic search
  - Image embedding and captioning

- 🧠 **Advanced Retrieval**:
  - Semantic search across text and image content
  - CLIP-based multimodal embeddings
  - Flexible query matching for text and image descriptions

- 🤖 **AI-Powered Analysis**:
  - LLaVA model integration via Ollama for multimodal understanding
  - Context-aware response generation
  - Ability to reason over both textual and visual information

## System Architecture

### Components
1. **Data Ingestion Pipeline**
   - MongoDB for document storage
   - RabbitMQ for message queuing
   - Bytewax for stream processing
   - Qdrant for vector storage

2. **Retrieval System**
   - Sentence Transformers for text embeddings
   - OpenCLIP for multimodal embeddings
   - Advanced retrieval strategies

3. **Inference Engine**
   - LLaVA model via Ollama for multimodal reasoning
   - Context-aware prompt engineering

### Technology Stack
- Python 3.10+
- MongoDB
- RabbitMQ
- Qdrant Vector Database
- Sentence Transformers
- OpenCLIP
- Ollama
- Bytewax
- FastAPI
- Streamlit

## Getting Started

### Prerequisites
- Docker
- Docker Compose
- Python 3.10+
- Ollama (for LLaVA model)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/multimodal-rag-system.git
cd multimodal-rag-system
```

2. Install Ollama and pull LLaVA model
```bash
ollama pull llava
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your specific configurations
```

4. Build and run with Docker Compose
```bash
docker-compose up --build
```

## Usage

### PDF Processing
1. Upload PDFs through the Streamlit UI
2. System automatically:
   - Extracts text and images
   - Processes and embeds content
   - Stores in vector database

### Querying
- Ask questions about your documents
- Supports text-only and multimodal queries
- Example queries:
  ```
  "Explain the architecture diagram in the document"
  "What are the main components of this system?"
  "Describe the workflow shown in the images"
  ```

## Advanced Configuration

### Embedding Models
- Configurable embedding models in `config.py`
- Currently uses:
  - Sentence Transformers for text
  - OpenCLIP for multimodal embeddings

### Retrieval Strategies
- Configurable top-k results
- Flexible text and image search
- Reranking capabilities

## Performance Optimization

- Caching of embedding models
- Efficient vector storage in Qdrant
- Streaming data processing with Bytewax
- GPU acceleration support

## Roadmap
- [ ] Add more advanced image understanding models
- [ ] Implement more sophisticated multimodal fusion
- [ ] Enhance prompt engineering techniques
- [ ] Add support for more document types

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - [Your Email]

Project Link: [https://github.com/yourusername/multimodal-rag-system](https://github.com/yourusername/multimodal-rag-system)

---

**Note**: This system is a research prototype. Always validate critical information from primary sources.