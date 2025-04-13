
from db import QdrantDatabaseConnector
# from qdrant_client import models
from sentence_transformers.SentenceTransformer import SentenceTransformer

import utils
from config import settings


from utils.logging import get_logger

logger = get_logger(__name__)




class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and Multitenancy search.
    """

    def __init__(self, query: str) -> None:
        self._client = QdrantDatabaseConnector()
        self.query = query
        self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL_ID)


    def _search_single_query(
        self, query: str, k: int):
        assert k > 3, "k should be greater than 3"

        query_vector = self._embedder.encode(query).tolist()

        vectors = [
            self._client.search(
                collection_name="vector_pdfs",
                query_vector=query_vector,
                limit=k // 3,
            )
        ]

        return utils.flatten(vectors)

    def retrieve_top_k(self, k: int) -> list:
        hits = self._search_single_query(self.query, k)
        logger.info("All documents retrieved successfully.", num_documents=len(hits))

        return hits 

    def set_query(self, query: str):
        self.query = query
        
        