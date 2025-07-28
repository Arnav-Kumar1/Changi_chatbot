# streamlit_app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
API_ENDPOINT = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/qa")

st.set_page_config(page_title="🛫 Changi & Jewel Chatbot", layout="wide")
st.title("🛫 Changi & Jewel Airport RAG Chatbot")
st.markdown("Ask anything about Changi or Jewel websites using a RAG-powered assistant.")

# --- Sidebar for API Key input ---
st.sidebar.header("🔐 API Key")
user_api_key = st.sidebar.text_input("Enter your Gemini API Key:", value=API_KEY, type="password")
if not user_api_key:
    st.sidebar.warning("API key is required to query Gemini.")
    st.stop()

# --- User Query ---
user_query = st.text_input("💬 What would you like to know?")
ask_button = st.button("Ask")

# --- Query the RAG API ---
if ask_button:
    if not user_query.strip():
        st.warning("❗ Please enter a question.")
    else:
        with st.spinner("🤖 Generating answer..."):
            try:
                res = requests.post(
                    API_ENDPOINT,
                    json={"user_query": user_query, "api_key": user_api_key},
                    timeout=45
                )

                if res.status_code == 200:
                    result = res.json()
                    st.success("✅ Answer:")
                    st.write(result.get("answer", "No answer returned."))

                    sources = result.get("sources", [])
                    if sources:
                        st.markdown("### 📎 Sources:")
                        for url in sources:
                            st.markdown(f"- [{url}]({url})")
                    else:
                        st.info("ℹ️ No sources provided.")
                else:
                    st.error(f"❌ {res.status_code} - {res.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try again later.")
            except Exception as e:
                st.error(f"🚨 Unexpected error: {str(e)}")
