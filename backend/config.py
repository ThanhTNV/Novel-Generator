from pathlib import Path
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    default_llm_provider: str = "claude"
    default_model: str = "claude-sonnet-4-20250514"

    # Dense embeddings for Zero-Mem's semantic view. "sentence-transformers"
    # (bge-m3, multilingual) when installed, "openai", or "hash" for the
    # dependency-free fallback. Retrieval degrades gracefully without a model:
    # BM25 + the entity graph carry it.
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-m3"

    # Zero-Mem substrate (SQLite). Replaces the old ChromaDB vectorstore.
    zero_mem_db: str = str(ROOT_DIR / "data" / "zero_mem.db")

    # Optional Gemini SLM extractor for Zero-Mem: adds relation triples and
    # unlisted entities on top of the token-free gazetteer/pattern NER.
    # Enabled automatically when GEMINI_API_KEY (or GOOGLE_API_KEY) is set;
    # set ZERO_MEM_EXTRACTOR=local to force it off.
    gemini_api_key: str = ""
    google_api_key: str = ""
    zero_mem_extractor: str = "auto"   # auto | gemini | local
    zero_mem_extract_model: str = "gemini-2.5-flash-lite"

    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    rules_dir: str = str(ROOT_DIR / "rules")
    skills_dir: str = str(ROOT_DIR / "skills")
    prompts_dir: str = str(ROOT_DIR / "prompts")
    context_dir: str = str(ROOT_DIR / "context")
    chapters_dir: str = str(ROOT_DIR / "chapters")

    # Generation defaults
    default_chunk_size: int = 400
    default_chunk_overlap: int = 80
    default_top_k: int = 5
    default_target_words: int = 2000
    max_context_tokens: int = 3000

    # Output budget. Vietnamese costs far more tokens per word than English —
    # measured 2.26 tok/word on this corpus with cl100k — so a flat 4096 cap
    # truncated a default 2000-word chapter mid-sentence. The generation
    # budget is derived from the requested length instead of hard-coded.
    output_tokens_per_word: float = 2.4
    output_token_headroom: float = 1.2
    min_output_tokens: int = 1024
    max_output_tokens: int = 16384

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}


settings = Settings()
