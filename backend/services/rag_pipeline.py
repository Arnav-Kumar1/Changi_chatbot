from services.vectorstore import vector_search
from services.embeddings import deduplicate_by_embedding
from services.data_loader import load_all_embedding_chunks
from sparse_search import SparseSearchIndex
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os

RETRIEVE_TOP_K = 50
RERANK_TOP_N = 20
FINAL_MAX_TOKENS = 3000
DUPLICATE_SIM_THRESHOLD = 0.9

# Load sparse index as before (sparse search uses raw chunks)
sparse_index = SparseSearchIndex([
    "data/changia_embedding_ready_raw_chunks.jsonl",
    "data/jewel_embedding_ready_raw_chunks.jsonl"
])

# LLM initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
    convert_system_message_to_human=True
)

# Module-level cached embedding lookup dictionaries
_embedding_lookup = None

def get_embedding_lookup():
    global _embedding_lookup
    if _embedding_lookup is None:
        # Unpack dense and sparse embedding chunks from data loader
        dense_chunks, sparse_chunks = load_all_embedding_chunks()
        # Combine all chunks for a comprehensive lookup
        all_chunks = dense_chunks + sparse_chunks
        _embedding_lookup = {
            c['chunk_id']: c.get('metadata', {}).get('embedding') or c.get('embedding')
            for c in all_chunks
            if (c.get('metadata', {}).get('embedding') or c.get('embedding')) is not None
        }
        print(f"[DEBUG] Loaded {_embedding_lookup.keys().__len__()} chunks into embedding lookup")
    return _embedding_lookup

def build_context(chunks: list) -> str:
    context_parts = []
    for c in chunks:
        text = c['metadata']['text']
        title = c['metadata'].get('title', '')
        url = c['metadata'].get('url', '')
        context_parts.append(f"{title} | Source: {url}\n{text}")
    return "\n\n".join(context_parts)

def trim_to_token_limit(chunks, max_tokens=FINAL_MAX_TOKENS):
    acc_tokens, selected = 0, []
    for c in chunks:
        token_count = len(c['metadata']['text'].split())
        if acc_tokens + token_count > max_tokens:
            break
        selected.append(c)
        acc_tokens += token_count
    return selected

def rerank(query: str, candidates: list, top_n=RERANK_TOP_N):
    # Simple placeholder: just truncate
    truncated = candidates[:top_n]
    print(f"[DEBUG][rerank] {len(truncated)} candidates, embedding presence:")
    for i, c in enumerate(truncated):
        emb = c.get('metadata', {}).get('embedding') or c.get('embedding')
        print(f"  candidate {i} embedding present? {'Yes' if emb else 'No'}")
    return truncated

def hybrid_retrieve(query: str, top_k=RETRIEVE_TOP_K):
    dense_results = vector_search(query, top_k=top_k)
    sparse_results = sparse_index.sparse_search(query, top_k=top_k)

    print(f"[DEBUG][hybrid_retrieve] Dense results count: {len(dense_results)}")
    dense_with_emb = sum(
        1 for c in dense_results
        if c.get('metadata', {}).get('embedding') or c.get('embedding')
    )
    print(f"[DEBUG][hybrid_retrieve] Dense chunks with embeddings: {dense_with_emb}")

    print(f"[DEBUG][hybrid_retrieve] Sparse results count: {len(sparse_results)}")
    sparse_with_emb = sum(
        1 for c in sparse_results
        if c.get('metadata', {}).get('embedding') or c.get('embedding')
    )
    print(f"[DEBUG][hybrid_retrieve] Sparse chunks with embeddings: {sparse_with_emb}")

    # Convert sparse results to the expected chunk dict shape, without embeddings yet
    sparse_candidates = [
        {
            'chunk_id': c.get('chunk_id', ''),
            'metadata': {
                'text': c.get('text', ''),
                'chunk_id': c.get('chunk_id', ''),
                'url': c.get('url', ''),
                'section': c.get('section', ''),
                'title': c.get('title', '')
                # embeddings will be added next
            }
        }
        for c in sparse_results
    ]

    # Enrich sparse candidates with embeddings from lookup
    emb_lookup = get_embedding_lookup()
    for c in sparse_candidates:
        chunk_id = c['chunk_id']
        embedding = emb_lookup.get(chunk_id)
        if embedding:
            c['metadata']['embedding'] = embedding
            c['embedding'] = embedding  # also attach at root for safety

    # Merge dense and enriched sparse candidates, deduplicate by URL on merge
    seen_urls = set()
    combined = []
    for c in dense_results + sparse_candidates:
        url = c['metadata'].get('url', '')
        if url and url not in seen_urls:
            combined.append(c)
            seen_urls.add(url)

    print(f"[DEBUG][hybrid_retrieve] Combined chunks count after dedup by URL: {len(combined)}")
    combined_with_emb = sum(
        1 for c in combined if c.get('metadata', {}).get('embedding') or c.get('embedding')
    )
    print(f"[DEBUG][hybrid_retrieve] Combined chunks WITH embeddings: {combined_with_emb}")
    print(f"[DEBUG][hybrid_retrieve] Combined chunks WITHOUT embeddings: {len(combined) - combined_with_emb}")

    return combined

def ask_llm(query: str, context: str) -> str:
    system_prompt = (
        "You are an expert assistant for Changi Airport and Jewel Changi Airport, "
        "tasked with providing the most accurate, concise, and helpful answers strictly based on the provided context. "
        "Use ONLY the information found in the context to answer the user’s question. "
        "Cite relevant URLs directly with phrases like ‘Learn more here: [URL]’. "
        "Never invent or speculate, and never reference that you are an AI model. "
        "If multiple sources support an answer, summarize clearly and cite all relevant URLs. "
        "If the context does not contain a direct answer, point the user to the most relevant URLs without fabricating information. "
        "Avoid phrases like ‘The provided context mentions’, ‘Information is not available’, or ‘I don’t know’. "
        "Do not add apologies or disclaimers. Respond professionally and politely.\n\n"

        "Use the following examples as guidance:\n\n"

        "**Example 1:**\n"
        "Q: How can I get from Terminal 2 to Jewel?\n"
        "A: You can walk from Terminal 2 to Jewel Changi Airport via the link bridges. Learn more here: https://www.changiairport.com/en/maps.html\n\n"

        "**Example 2:**\n"
        "Q: What are the opening hours of Jewel?\n"
        "A: Jewel is open daily from 10:00 AM to 10:00 PM. For detailed info, visit: https://www.jewelchangiairport.com/en/plan-your-visit/opening-hours.html\n\n"

        "**Example 3:**\n"
        "Q: Is there free Wi-Fi available?\n"
        "A: Yes, free Wi-Fi is available throughout Changi Airport terminals and Jewel. Learn more here: https://www.changiairport.com/en/airport-guide/wi-fi.html\n\n"

        "**What NOT to say:**\n"
        "‘The answer is not available in the context.’ or ‘According to the document, ...’ or ‘I don’t know.’\n"
        "Instead, directly point users towards relevant sources.\n\n"

        f"Context:\n{context}"
    )


    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {query}")
    ]

    return llm.invoke(messages).content.strip()

def rag_pipeline(user_query: str, api_key: str) -> dict:
    if not user_query or not api_key:
        raise ValueError("Missing user_query or api_key")
    os.environ["GOOGLE_API_KEY"] = api_key
    llm.google_api_key = api_key

    candidates = hybrid_retrieve(user_query)
    print(f"[DEBUG] Candidates count before rerank: {len(candidates)}")

    reranked = rerank(user_query, candidates)
    print(f"[DEBUG] Reranked count: {len(reranked)}")

    filtered = deduplicate_by_embedding(reranked, threshold=DUPLICATE_SIM_THRESHOLD)
    print(f"[DEBUG] Filtered (dedup) count: {len(filtered)}")

    top_chunks = trim_to_token_limit(filtered, max_tokens=FINAL_MAX_TOKENS)
    print(f"[DEBUG] Top chunks after token trim: {len(top_chunks)}")
    if top_chunks:
        print(f"[DEBUG] Sample top chunk metadata: {top_chunks[0].get('metadata')}")

    context = build_context(top_chunks)
    print(f"[DEBUG] Context length (chars): {len(context)}")
    if len(context) > 300:
        print(f"[DEBUG] Context preview:\n{context[:300]}")

    answer = ask_llm(user_query, context)
    print(f"[DEBUG] Answer from LLM: {answer}")

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
        "sources": sources[:2]
    }
