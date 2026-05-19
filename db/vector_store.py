from __future__ import annotations
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_postgres import PGVector
from config import Settings
from llm_factory import get_embeddings

class VectorStore:
    """thin wrapper around langchain's pgvector integration.
    handles document ingesion and similiarity based retreival"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = get_embeddings(settings)
        self.store =  None  #lazy inintialization on first use
        
    def _get_store(self) -> PGVector:
        """init pgvector on first call, reuse on subsequent calls.
            lazy iit means we dont connect to postgres at import time-
            only when we actually need it"""

        if self.store is None:
            self.store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.settings.collection_name,
                connection=self.settings.postgres_url,
                use_jsonb=True,
            )
        return self.store
    
    def add_documents(self, documents: List[Document]):
        """embed and persist a list of doc objects.
            returns the list of assigned UUIDs"""
        store = self._get_store()
        ids  =store.add_documents(documents)
        return ids
    
    def similiarity_search_with_score(self, query: str, k : int | None = None,) -> List[Tuple[Document, float]]:
        """embed the query and return the top k most similiar chunks
            each resukt is a (document, score) tuple where score is in [0, 1]
            higher score implies more similiarity"""
        k = k or self.settings.retrieval_k
        store  = self._get_store()
        
        #relv score
        results = store.similarity_search_with_score(query, k=k)
        return results
    
    def delete_collection(self) -> None:
        """drop the entire pgvector collcetion.
            useful for resetting during development"""
        store = self._get_store()
        store.delete_collection()
        self._store = None  #reset store so it will be reinitialized on next use
        
    def collection_status(self) -> dict:
        """quick health check - confirms the store is reachable"""
        try:
            self._get_store()
            return {
                "status": "ok",
                "collection": self.settings.collection_name,
                "postgres": self.settings.postgres_host,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}