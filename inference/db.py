from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch, Distance, VectorParams

from utils.logging import get_logger
from config import settings

logger = get_logger(__name__)
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch, Distance, VectorParams

from utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __init__(self) -> None:
        if self._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    self._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_APIKEY,
                    )
                else:
                    self._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                    url=settings.QDRANT_CLOUD_URL,
                )
                raise

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def create_non_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name, vectors_config={}
        )

    def create_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_SIZE,
                distance=Distance.COSINE
            ),
        )

    def create_collection(self, collection_name: str, is_vector: bool = True):
        """
        Wrapper to create either a vector or non-vector collection depending on use case.
        Default: vector collection.
        """
        if is_vector:
            self.create_vector_collection(collection_name)
        else:
            self.create_non_vector_collection(collection_name)
        logger.info(f"Created collection `{collection_name}` (vector={is_vector})")

    def write_data(self, collection_name: str, points: Batch):
        try:
            self._instance.upsert(collection_name=collection_name, points=points)
        except Exception:
            logger.exception("An error occurred while inserting data.")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self):
        if self._instance:
            self._instance.close()
            logger.info("Connected to database has been closed.")



class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __init__(self) -> None:
        if self._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    self._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_APIKEY,
                    )
                else:
                    self._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )
            except UnexpectedResponse:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                    url=settings.QDRANT_CLOUD_URL,
                )

                raise

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        return self._instance.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)
        
    def get_point(self, collection_name: str, point_id: str):
        """Get a specific point by ID"""
        try:
            return self._instance.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {str(e)}")
            return None

    def close(self):
        if self._instance:
            self._instance.close()
            logger.info("Connected to database has been closed.")