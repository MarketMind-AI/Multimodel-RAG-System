from bytewax.outputs import DynamicSink, StatelessSinkPartition
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch
from qdrant_client.http.models import Distance, VectorParams

from utils.logging import get_logger
from db import QdrantDatabaseConnector
from models.base import VectorDBDataModel
from config import settings

logger = get_logger(__name__)


class QdrantOutput(DynamicSink):
    """
    Bytewax class that facilitates the connection to a Qdrant vector DB.
    Inherits DynamicSink to create different sink sources 
    (e.g, vector and non-vector collections)
    """

    def __init__(self, connection: QdrantDatabaseConnector, sink_type: str):
        self._connection = connection
        self._sink_type = sink_type

        # Configurations for collections to create
        collections_to_create = [
            {"name": "cleaned_pdfs", "is_vector": False},
            {"name": "vector_pdfs", "is_vector": True, "size": settings.EMBEDDING_SIZE},
            {"name": "cleaned_images", "is_vector": False},
            {"name": "vector_images", "is_vector": True, "size": 512}  # CLIP uses 512-dim embeddings
        ]

        # Create collections if they don't exist
        for collection in collections_to_create:
            try:
                # First try to get the collection to see if it exists
                try:
                    self._connection.get_collection(collection_name=collection['name'])
                    logger.info(f"Collection {collection['name']} already exists.")
                except UnexpectedResponse:
                    logger.info(f"Couldn't access the collection. Creating a new one...", collection_name=collection['name'])
                    
                    # Direct creation using the Qdrant client instance
                    if collection.get('is_vector', False):
                        size = collection.get('size', settings.EMBEDDING_SIZE)
                        self._connection._instance.create_collection(
                            collection_name=collection['name'],
                            vectors_config=VectorParams(
                                size=size,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(f"Created vector collection {collection['name']} with size {size}")
                    else:
                        self._connection._instance.create_collection(
                            collection_name=collection['name'],
                            vectors_config={}
                        )
                        logger.info(f"Created non-vector collection {collection['name']}")
            except Exception as e:
                logger.error(f"Error with collection {collection['name']}: {str(e)}")

    def build(self, worker_index: int, worker_count: int) -> StatelessSinkPartition:
        if self._sink_type == "clean":
            return QdrantCleanedDataSink(connection=self._connection)
        elif self._sink_type == "vector":
            return QdrantVectorDataSink(connection=self._connection)
        else:
            raise ValueError(f"Unsupported sink type: {self._sink_type}")


class QdrantCleanedDataSink(StatelessSinkPartition):
    def __init__(self, connection: QdrantDatabaseConnector):
        self._client = connection

    def write_batch(self, items: list[VectorDBDataModel]) -> None:
        if not items:
            return

        payloads = [item.to_payload() for item in items]
        ids, data = zip(*payloads)
        collection_name = get_clean_collection(data_type=data[0]["type"])
        
        try:
            self._client.write_data(
                collection_name=collection_name,
                points=Batch(ids=ids, vectors={}, payloads=data),
            )

            logger.info(
                "Successfully inserted requested cleaned point(s)",
                collection_name=collection_name,
                num=len(ids),
            )
        except Exception as e:
            logger.error(
                f"Error inserting cleaned points to {collection_name}: {str(e)}",
                num=len(ids)
            )


class QdrantVectorDataSink(StatelessSinkPartition):
    def __init__(self, connection: QdrantDatabaseConnector):
        self._client = connection

    def write_batch(self, items: list[VectorDBDataModel]) -> None:
        if not items:
            return

        payloads = [item.to_payload() for item in items]
        ids, vectors, meta_data = zip(*payloads)
        collection_name = get_vector_collection(data_type=meta_data[0]["type"])
        
        # Validate vector dimensions
        if collection_name == "vector_images":
            expected_size = 512  # CLIP embeddings
        else:
            expected_size = settings.EMBEDDING_SIZE  # Text embeddings
            
        # Log vector dimension check
        for i, vector in enumerate(vectors):
            actual_size = len(vector)
            if actual_size != expected_size:
                logger.warning(
                    f"Vector dimension mismatch for ID {ids[i]}: "
                    f"expected {expected_size}, got {actual_size}. "
                    "This may cause insert errors.",
                    collection_name=collection_name,
                )
        
        try:
            self._client.write_data(
                collection_name=collection_name,
                points=Batch(ids=ids, vectors=vectors, payloads=meta_data),
            )

            logger.info(
                "Successfully inserted requested vector point(s)",
                collection_name=collection_name,
                num=len(ids),
            )
        except Exception as e:
            logger.error(
                f"Error inserting vectors into {collection_name}: {str(e)}",
                num=len(ids),
                first_id=ids[0] if ids else None,
            )


def get_clean_collection(data_type: str) -> str:
    if data_type == "pdf_documents":
        return "cleaned_pdfs"
    elif data_type == "image_documents":
        return "cleaned_images"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_vector_collection(data_type: str) -> str:
    if data_type == "pdf_documents":
        return "vector_pdfs"
    elif data_type == "image_documents":
        return "vector_images"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")