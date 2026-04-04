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

    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-m3"

    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    chroma_persist_dir: str = str(ROOT_DIR / "vectorstore")

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

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}


settings = Settings()
