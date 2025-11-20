# 🚀 FedSearch-NLP: Federated RAG QA System

A production-ready FastAPI backend for enterprise document search and question answering using Retrieval-Augmented Generation (RAG).

## 📋 Features

✅ **RAG Pipeline**: Combines document retrieval with LLM-based answer generation  
✅ **FAISS Vector Search**: Fast semantic document search  
✅ **Enterprise Ready**: Built with FastAPI for production use  
✅ **Sample Data Included**: Pre-loaded with company documents  
✅ **Interactive API Docs**: Auto-generated Swagger UI  
✅ **Extensible**: Easy to add federated learning capabilities

---

## 🏗️ Project Structure

```
fedsearch_nlp/
├── app/
│   ├── main.py                 # FastAPI app entry
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── models.py           # Request/response models
│   ├── core/
│   │   ├── config.py           # Configuration
│   │   └── rag_engine.py       # RAG orchestration
│   └── services/
│       ├── document_processor.py  # Document loading
│       ├── retriever.py           # FAISS retrieval
│       └── generator.py           # Answer generation
├── data/
│   ├── company_docs/           # 📄 Company documents (YOUR DATA)
│   │   ├── hr_policy.txt
│   │   ├── it_sop.txt
│   │   ├── legal_doc.txt
│   │   └── product_guide.txt
│   └── embeddings/             # Generated FAISS indices
├── requirements.txt
├── .env
└── README.md
```

---

## ⚡ Quick Start (3 Steps)

### Option 1: Automated Setup (Recommended)

```bash
# 1. Make setup script executable
chmod +x setup_and_run.sh

# 2. Run setup
./setup_and_run.sh

# 3. Start server
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

### Option 2: Manual Setup

```bash
# 1. Create directories
mkdir -p app/api app/core app/services app/utils
mkdir -p data/company_docs data/embeddings models

# 2. Create __init__.py files
touch app/__init__.py app/api/__init__.py app/core/__init__.py
touch app/services/__init__.py app/utils/__init__.py

# 3. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Create company documents (see data creation script)
python create_company_docs.py

# 5. Start server
python -m uvicorn app.main:app --reload
```

---

## 🎯 Usage

### 1. Access API Documentation

Open your browser: **http://localhost:8000/docs**

### 2. Index Documents (First Time)

```bash
curl -X POST "http://localhost:8000/api/index" \
  -H "Content-Type: application/json" \
  -d '{"reindex": false}'
```

**Response:**
```json
{
  "status": "success",
  "documents_indexed": 42,
  "message": "Index built successfully"
}
```

### 3. Query the System

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual leave policy?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "answer": "The annual leave policy provides 20 days per year for full-time employees.",
  "retrieved_documents": [
    {
      "content": "Annual Leave: 20 days per year...",
      "score": 0.89,
      "source": "hr_policy.txt"
    }
  ],
  "confidence": 0.92
}
```

### 4. Check System Health

```bash
curl http://localhost:8000/api/health
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API overview |
| GET | `/api/health` | System health check |
| POST | `/api/index` | Index/reindex documents |
| POST | `/api/query` | Ask questions |
| GET | `/api/documents/stats` | Document statistics |

---

## 🧪 Test Queries

Try these sample questions:

```bash
# HR Policy
"How many sick leave days do employees get?"
"What is the remote work policy?"
"When are salary reviews conducted?"

# IT Procedures
"What is the password policy?"
"How do I report a security incident?"
"How often are backups performed?"

# Legal
"What is the data retention policy?"
"How long is the non-compete clause?"

# Products
"What is the pricing for CloudSync Pro?"
"Which products support SSO?"
"What compliance certifications do we have?"
```

---

## 📄 Company Documents (Data Files)

The system includes 4 sample company documents in `data/company_docs/`:

1. **hr_policy.txt** - HR policies (leave, benefits, working hours)
2. **it_sop.txt** - IT procedures (access, backups, security)
3. **legal_doc.txt** - Legal guidelines (IP, compliance, contracts)
4. **product_guide.txt** - Product information (pricing, features)

### Adding Your Own Documents

1. Place `.txt` files in `data/company_docs/`
2. Reindex documents:
   ```bash
   curl -X POST "http://localhost:8000/api/index" \
     -H "Content-Type: application/json" \
     -d '{"reindex": true}'
   ```

---

## 🔧 Configuration

Edit `.env` file to customize:

```env
# Models
RETRIEVER_MODEL="sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL="google/flan-t5-base"

# Performance
TOP_K=3
MAX_LENGTH=512

# Paths
COMPANY_DOCS_PATH="data/company_docs"
EMBEDDINGS_PATH="data/embeddings"
```

---

## 🚀 Production Deployment

### Docker (Coming Soon)

```bash
docker build -t fedsearch-nlp .
docker run -p 8000:8000 fedsearch-nlp
```

### Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/fedsearch.service

# Add configuration (see docs)
sudo systemctl enable fedsearch
sudo systemctl start fedsearch
```

---

## 🧠 How It Works

1. **Document Processing**: Text files are split into semantic chunks
2. **Embedding**: Each chunk is converted to a 384-dim vector
3. **Indexing**: FAISS creates a searchable vector database
4. **Retrieval**: User query → vector → find top-K similar chunks
5. **Generation**: Flan-T5 generates answer from retrieved context

---

## 🛠️ Tech Stack

- **FastAPI** - Web framework
- **Sentence-Transformers** - Document embeddings
- **FAISS** - Vector similarity search
- **Transformers** - Flan-T5 for answer generation
- **PyTorch** - Deep learning backend

---

## 📊 Performance

- **Indexing**: ~50 documents/second
- **Query Latency**: ~500ms (CPU), ~200ms (GPU)
- **Memory**: ~2GB (models + indices)

---

## 🔮 Future Enhancements

- [ ] Federated Learning integration
- [ ] Multi-client architecture
- [ ] Differential privacy (DP-SGD)
- [ ] PDF/DOCX support
- [ ] User authentication
- [ ] Chat history
- [ ] Streaming responses

---

## 🐛 Troubleshooting

### Models not downloading?

Set cache directory:
```bash
export TRANSFORMERS_CACHE="./models"
export SENTENCE_TRANSFORMERS_HOME="./models"
```

### Out of memory?

Use smaller models in `.env`:
```env
RETRIEVER_MODEL="sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL="google/flan-t5-small"
```

### Port already in use?

Change port in `.env`:
```env
PORT=8001
```

---

## 📝 License

MIT License - See LICENSE file

---

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

---

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: GitHub Issues
- 📚 Docs: http://localhost:8000/docs

---

**Built with ❤️ for enterprise document search**