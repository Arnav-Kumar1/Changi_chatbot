# 🛬 Changi & Jewel Airport RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot for answering queries related to **Changi Airport** and **Jewel Changi** using:

- 🔍 Dense vector search via **Pinecone**
- 🧠 Sparse keyword-based TF-IDF search
- 🤖 **Google Gemini LLM** (via **LangChain**)
- 🛠️ Fullstack architecture with **FastAPI (backend)** and **Streamlit (frontend)**
- 🔐 **BYOK (Bring Your Own Key)** supported — users can supply their own Google API key

---

## ⚙️ Project Setup & Usage

This project supports **local development**, **Docker-based environments**, and **cloud deployment**.

- 💻 Local setup (recommended for fast iteration)
- 🐳 Dockerized setup (recommended for portability and isolation)
- ☁️ Cloud deployment via **Render** and **Streamlit Cloud**

---

## 🧪 1. Local Setup

### ✅ Prerequisites

- Python 3.10+
- Git (if cloning the repo)
- A terminal that supports virtual environments

### 🛠️ Setup Instructions

```bash
# 📦 OPTION 1: Clone the repo (recommended)
git clone https://github.com/<your-org>/changi_jewel_rag_chatbot.git
cd changi_jewel_rag_chatbot

# OR

# 📁 OPTION 2: Download ZIP from GitHub
# - Go to the repo page → Click "Code" → "Download ZIP"
# - Extract the folder
# - Then navigate into the extracted directory:
cd changi_jewel_rag_chatbot-main
```

### 🔐 Create a `.env` file in the project root (see Environment Variables section below)

---

### ⚙️ Backend Setup

```bash
# Move into backend directory
cd backend

# Create a virtual environment
python -m venv chatbot_env

# Activate it
# Windows:
chatbot_env\Scripts\activate.bat
# Mac/Linux:
source chatbot_env/bin/activate

# Install backend dependencies
pip install -r requirements_backend.txt
```

### 📥 Preprocessing & Embedding

- Run scripts `1_*.py` through `7_*.py` in the `scripts/` folder.
- These will scrape, clean, chunk, embed data, and save vectors to Pinecone.
- After execution, place the following files inside `backend/data/`:
  - `Google_changia_sparse_embs.jsonl`
  - `Google_jewel_sparse_embs.jsonl`

### 🚀 Launch Backend

```bash
# From inside backend/ with virtual env active
uvicorn main:app
```

---

### 🎨 Frontend Setup

```bash
# Go back to root
cd ..

# Install frontend requirements
pip install -r requirements_frontend.txt

# Run Streamlit app
streamlit run frontend/frontend.py
```

---

## 🐳 2. Dockerized Setup

### ✅ Prerequisites

- Docker
- Docker Compose

### 🧪 Quickstart

```bash
# Ensure your embedding .jsonl files are in backend/data/
docker-compose up --build
```

- 📡 Backend available at: [http://localhost:8000/docs](http://localhost:8000/docs)  
- 🖥️ Frontend available at: [http://localhost:8501](http://localhost:8501)  
- 🔁 Live reload supported during development

---

## ☁️ 3. Cloud Deployment

### 🚀 Deploy Backend to Render

1. Push your backend code (with Dockerfile) to GitHub
2. Create a **Web Service** on [Render](https://render.com):
   - Environment: **Docker**
   - Root Directory: `backend`
   - Dockerfile Path: `backend/Dockerfile.backend`
3. Add environment variables via Render dashboard
4. Make sure `backend/data/*.jsonl` files are included in the repo or uploaded manually

### 🚀 Deploy Frontend

1. Push `frontend.py` and `requirements.txt` to GitHub
2. Deploy using [Streamlit Cloud](https://streamlit.io/cloud):
   - Add **Secrets**: `BACKEND_API_URL`, `GOOGLE_API_KEY`
3. ✅ Alternatively, deploy frontend via Render (Dockerized)

---

## 🔗 Connecting Frontend to Backend

In Streamlit secrets or your `.env`, set:

```env
BACKEND_API_URL=https://<your-backend-url>/api/qa
```

---

## 🔐 Environment Variables

Set the following in a `.env` file or deployment environment:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX_NAME=your_index_name
BACKEND_API_URL=http://localhost:8000/api/qa  # For local frontend
```

---

## 📡 REST API Reference

- Endpoint: `POST /api/qa`

### 🔍 Sample Request

```json
{
  "user_query": "How do I get to Changi Airport Terminal 3?",
  "api_key": "optional_google_api_key"
}
```

- Returns: LLM-generated answer and source links

---

## 🧯 Troubleshooting

- ✅ Ensure `.jsonl` embedding files exist inside `backend/data/`
- 🔑 Confirm Pinecone API and Google Gemini API keys are valid
- 📊 Check API usage quotas
- 🔍 Use Swagger UI at [localhost:8000/docs](http://localhost:8000/docs)
- 🧠 Enable logging in `rag_pipeline.py`, `vectorstore.py` for debugging

---

## 🤝 Contributing

1. Fork this repo
2. Clone your fork
3. Set up the project locally (see above)
4. Make improvements with clear commits
5. Open a Pull Request

---

## 📜 License & Contact

Refer to [`LICENSE`](./LICENSE) or contact the maintainers for support and licensing.

---

**Thanks for using the Changi & Jewel Airport RAG Chatbot!**  
✈️ *Happy coding and safe travels!*
