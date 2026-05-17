from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache

class Settings(BaseSettings):
    
    #llm provider--------------------
    llm_provider: Literal['openai', 'groq', 'ollama]'] = Field(default="groq")
    
    #openai-------------------------
    openai_api__key : str = Field(default="")
    openai_model : str = Field(default= "gpt-4o-mini")
    
    #groq---------------------------
    groq_api_key : str =Field(default="")
    groq_model : str = Field(default="llama-3.3-70b-instruct")
    
    #ollama-------------------------
    ollama_base_url : str = Field(default="http://localhost:11434")
    ollama_model : str = Field(default="llama-3.2")
    
    
    #embdedding-----------------------
    embedding_model : str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_data : int =Field(default=384)
    
    #postgres----pgvector----------------------
    postgres_host : str = Field(default="localhost")
    postgres_port : int = Field(default=5432)
    postgres_db : str = Field(default="self_rag")
    postgres_user : str = Field(default="selfrag")
    postgres_password : str = Field(default="selfrag123")
    
    @property
    def postgres_url(self) -> str :
        return f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    #retrieval----------------------
    chunk_size : int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    retrieval_k : int = Field(default=5)
    relevance_score_threshold: float = Field(default=0.6)
    max_rewrite_attempts: int =Field(default=3)
    
    #flask----------------------
    flask_host: str =Field(default="0.0.0.0")
    flask_port: int = Field(default=5000)
    flask_debug : bool = Field(default=False)
    
    #collection name in pgvector-----
    collection_name: str  = Field(default="documents")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()