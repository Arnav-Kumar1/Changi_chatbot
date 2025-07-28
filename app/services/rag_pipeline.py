import os
from app.services.vectorstore import vector_search
from app.services.embeddings import query_encoder, reranker, deduplicate_by_embedding
from app.sparse_search import SparseSearchIndex
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# --- Configs ---
RETRIEVE_TOP_K = 50
RERANK_TOP_N = 20
FINAL_MAX_TOKENS = 3000
DUPLICATE_SIM_THRESHOLD = 0.9

# --- Initialize Sparse Search Index ---
sparse_index = SparseSearchIndex([
    "data/changia_embedding_ready_raw_chunks.jsonl",
    "data/jewel_embedding_ready_raw_chunks.jsonl"
])

# --- Gemini LLM Client (lazy key override later) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
    convert_system_message_to_human=True
)

# --- Context Builder ---
def build_context(chunks: list) -> str:
    context_parts = []
    for c in chunks:
        text = c['metadata']['text']
        title = c['metadata'].get('title', '')
        url = c['metadata'].get('url', '')
        context_parts.append(f"{title} | Source: {url}\n{text}")
    return "\n\n".join(context_parts)

# --- Token Limit Trimmer ---
def trim_to_token_limit(chunks, max_tokens=FINAL_MAX_TOKENS):
    acc_tokens, selected = 0, []
    for c in chunks:
        token_count = len(c['metadata']['text'].split())
        if acc_tokens + token_count > max_tokens:
            break
        selected.append(c)
        acc_tokens += token_count
    return selected

# --- Reranking ---
def rerank(query: str, candidates: list, top_n=RERANK_TOP_N):
    if not candidates:
        return []
    texts = [
        f"[{c['metadata'].get('section','General')}] {c['metadata'].get('title','')}: {c['metadata']['text']}"
        for c in candidates
    ]
    query_pairs = [(query, t) for t in texts]
    scores = reranker.predict(query_pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [item[0] for item in ranked]

# --- Hybrid Retrieval ---
def hybrid_retrieve(query: str, top_k=RETRIEVE_TOP_K):
    dense_results = vector_search(query, top_k=top_k)
    sparse_results = sparse_index.sparse_search(query, top_k=top_k)

    sparse_candidates = [
        {
            'metadata': {
                'text': c.get('text', ''),
                'chunk_id': c.get('chunk_id', ''),
                'url': c.get('url', ''),
                'section': c.get('section', ''),
                'title': c.get('title', '')
            }
        }
        for c in sparse_results
    ]

    seen_urls = set()
    combined = []
    for c in dense_results + sparse_candidates:
        url = c['metadata'].get('url', '')
        if url and url not in seen_urls:
            combined.append(c)
            seen_urls.add(url)
    return combined

# --- LLM Call ---
def ask_llm(query: str, context: str) -> str:
    system_prompt = (
        "You are a helpful and professional assistant for Changi Airport and Jewel. "
        "Use ONLY the provided context to answer the user's question. "
        "If the answer is in the context, cite it clearly. If there are relevant URLs, include them directly in the response. "
        "Do NOT invent or mention missing information. "
        "Avoid vague phrases like 'link provided in context'. Instead, say something like: "
        "'You can learn more here: https://example.com'. "
        "Be concise, factual, and polite.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {query}")
    ]

    return llm.invoke(messages).content.strip()

# --- Final Pipeline ---
def rag_pipeline(user_query: str, api_key: str) -> dict:
    if not user_query or not api_key:
        raise ValueError("Missing user_query or api_key")

    os.environ["GOOGLE_API_KEY"] = api_key
    llm.google_api_key = api_key

    candidates = hybrid_retrieve(user_query)
    reranked = rerank(user_query, candidates)
    filtered = deduplicate_by_embedding(reranked, model=query_encoder, threshold=DUPLICATE_SIM_THRESHOLD)
    top_chunks = trim_to_token_limit(filtered, max_tokens=FINAL_MAX_TOKENS)

    context = build_context(top_chunks)
    answer = ask_llm(user_query, context)

    # Select only URLs whose chunk content overlaps with answer text
    sources, seen_urls = [], set()
    answer_words = set(answer.lower().split())

    for chunk in top_chunks:
        url = chunk['metadata'].get('url', '')
        chunk_words = set(chunk['metadata']['text'].lower().split())
        if url and url not in seen_urls and answer_words & chunk_words:
            sources.append(url)
            seen_urls.add(url)

    return {
        "question": user_query,
        "answer": answer,
        "sources": sources[:3]  # Limit to top 3 unique, relevant sources
    }

