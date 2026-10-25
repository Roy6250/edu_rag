"""Central configuration. Replace the dummy strings with real values before running."""

# --- Credentials (dummy placeholders) ---
OPENAI_API_KEY = "sk-your-openai-api-key"
MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"

# --- MongoDB Atlas ---
DB_NAME = "edu_rag"
COLLECTION_NAME = "nodes"
VECTOR_INDEX_NAME = "vector_index"

# --- Models ---
# Multilingual BERT-family sentence encoder (768-d), standing in for mBERT from the paper.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM = 768
LLM_MODEL = "gpt-4o-mini"       # generator
EVAL_LLM_MODEL = "gpt-4o-mini"  # RAGAS judge

# --- Chunking / retrieval ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 1  # paper uses k=1

# Prompt template from the paper (Methods > Retrieval).
QA_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the answer, just say "
    "that you don't know. Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)

QGEN_PROMPT = (
    "You are an experienced teacher writing exam questions.\n"
    "Using ONLY the context below, write {n} {qtype} questions of {difficulty} difficulty "
    "on the topic \"{topic}\". Each question must have a correct answer that can be "
    "derived from the context.\n"
    "{format_hint}\n"
    "Return ONLY a JSON list, no extra text.\n\n"
    "Context:\n{context}\n"
)

MCQ_FORMAT = 'Format: [{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "B", "explanation": "..."}]'
SHORT_FORMAT = 'Format: [{"question": "...", "answer": "..."}]'
