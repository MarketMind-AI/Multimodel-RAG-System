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
                logger.info(f"Attempting to connect to Qdrant with host: {settings.QDRANT_DATABASE_HOST}")
                logger.info(f"Qdrant port: {settings.QDRANT_DATABASE_PORT}")
                
                self._instance = QdrantClient(
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                )
                
                # Perform a simple health check
                try:
                    collections = self._instance.get_collections()
                    logger.info(f"Successfully connected. Existing collections: {collections}")
                except Exception as health_check_error:
                    logger.error(f"Health check failed: {str(health_check_error)}")
                    
            except Exception as e:
                logger.error(f"Qdrant connection failed: {str(e)}")
                raise

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def create_collection(
        self, 
        collection_name: str, 
        is_vector: bool = True, 
        vector_size: int = None
    ):
        """
        Create a collection with flexible configuration
        
        Args:
            collection_name: Name of the collection to create
            is_vector: Whether to create a vector collection
            vector_size: Size of vectors (defaults to embedding size if not specified)
        """
        try:
            # Check if collection already exists
            try:
                self._instance.get_collection(collection_name=collection_name)
                logger.info(f"Collection {collection_name} already exists.")
                return
            except UnexpectedResponse:
                # Collection doesn't exist, so we'll create it
                pass

            # Create vector or non-vector collection
            if is_vector:
                # Use provided size or default to embedding size
                size = vector_size or settings.EMBEDDING_SIZE
                
                # Adjust size for image collections to 512
                if collection_name == "vector_images":
                    size = 512

                logger.info(f"Creating vector collection {collection_name} with size {size}")
                self._instance.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=size,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Creating non-vector collection {collection_name}")
                self._instance.create_collection(
                    collection_name=collection_name,
                    vectors_config={}
                )
            
            logger.info(f"Successfully created collection {collection_name}")
        
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            raise

    def create_non_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name, vectors_config={}
        )

    def create_vector_collection(self, collection_name: str, size: int = None):
        """
        Create a vector collection with configurable embedding size
        
        Args:
            collection_name: Name of the collection to create
            size: Dimension of the embedding vector. Defaults to text embedding size.
        """
        if size is None:
            size = settings.EMBEDDING_SIZE
        
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=size,
                distance=Distance.COSINE
            ),
        )

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