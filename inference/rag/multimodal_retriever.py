"""
Multimodal Retriever
------------------
A retriever that can search for both text and images using CLIP and text embeddings.
"""
import base64
import os
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from PIL import Image

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import flatten
from utils.logging import get_logger
from utils.clip_embeddings import generate_clip_text_embedding
from config import settings

from db import QdrantDatabaseConnector

logger = get_logger(__name__)


class MultimodalRetriever:
    """
    Retriever for multimodal search (both text and images)
    """

    def __init__(self, query: str, image_query: Optional[str] = None):
        """
        Initialize the multimodal retriever
        
        Args:
            query: Text query
            image_query: Optional image description query
        """
        self._client = QdrantDatabaseConnector()
        self.query = query
        self.image_query = image_query or query  # Use main query if no specific image query
        self._text_embedder = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
        
        # Initialize MongoDB connection for image retrieval
        mongo_uri = os.getenv(
            "MONGODB_URI",
            "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
        )
        self._mongo_client = MongoClient(mongo_uri)
        self._mongo_db = self._mongo_client["production"]
        
        logger.info(f"Initialized MultimodalRetriever with query: {query}")
        if image_query:
            logger.info(f"Image query: {image_query}")

    def _convert_point(self, point: Any, is_image: bool = False) -> Dict[str, Any]:
        """
        Convert a search result (ScoredPoint) to a dictionary.
        
        Args:
            point: The search result object.
            is_image: Flag indicating if this is an image result.
            
        Returns:
            Dictionary with standardized keys.
        """
        payload = getattr(point, "payload", {}) or {}
        if is_image:
            return {
                "id": getattr(point, "id", ""),
                "score": getattr(point, "score", 0.0),
                "source": payload.get("source", ""),
                "type": payload.get("type", "image_documents"),
                "image_id": payload.get("image_id", ""),
                "caption": payload.get("caption", None),
                "page_num": payload.get("page_num", None),
                "format": payload.get("format", None)
            }
        else:
            return {
                "id": getattr(point, "id", ""),
                "score": getattr(point, "score", 0.0),
                "source": payload.get("source", ""),
                "type": payload.get("type", "pdf_documents"),
                "content": payload.get("content", ""),
                "chunk_id": payload.get("id", ""),
                "page_range": payload.get("page_num", None)
            }

    def _search_text(self, query: str, k: int = 5) -> List[Any]:
        """
        Search for text content using text embeddings.
        
        Args:
            query: Text query.
            k: Number of results to return.
            
        Returns:
            List of raw text search results.
        """
        if k <= 0:
            return []

        query_vector = self._text_embedder.encode(query).tolist()

        try:
            vector_results = self._client.search(
                collection_name="vector_pdfs",
                query_vector=query_vector,
                limit=k,
            )
            logger.info(f"Retrieved {len(vector_results)} text results for query: '{query}'")
            return vector_results
        except Exception as e:
            logger.error(f"Error searching text vectors: {str(e)}")
            return []

    def _search_images(self, query: str, k: int = 3) -> List[Any]:
        """
        Search for image content using CLIP text embeddings.
        
        Args:
            query: Image description query.
            k: Number of results to return.
            
        Returns:
            List of raw image search results.
        """
        if k <= 0:
            return []

        try:
            query_vector = generate_clip_text_embedding(query).tolist()
            vector_results = self._client.search(
                collection_name="vector_images",
                query_vector=query_vector,
                limit=k,
            )
            logger.info(f"Retrieved {len(vector_results)} image results for query: '{query}'")
            return vector_results
        except Exception as e:
            logger.error(f"Error searching image vectors: {str(e)}")
            return []

    def _get_image_from_mongodb(self, image_id: str) -> Optional[bytes]:
        """
        Retrieve the actual image data from MongoDB.
        
        Args:
            image_id: The ID of the image to retrieve.
            
        Returns:
            Optional binary image data.
        """
        try:
            image_doc = self._mongo_db["image_documents"].find_one({"image_id": image_id})
            if image_doc and "image_data" in image_doc:
                logger.info(f"Found image data for image_id: {image_id}")
                return image_doc["image_data"]
            else:
                logger.warning(f"No image data found for image_id: {image_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving image data from MongoDB: {str(e)}")
            return None

    def retrieve_top_k(self, k_text: int = 5, k_images: int = 3) -> Dict[str, Any]:
        """
        Retrieve both text and image results and convert them to dictionaries.
        
        Args:
            k_text: Number of text results to return.
            k_images: Number of image results to return.
            
        Returns:
            Dictionary with keys "text_results" and "image_results".
        """
        text_raw = self._search_text(self.query, k_text)
        image_raw = self._search_images(self.image_query, k_images)

        text_results = [self._convert_point(point, is_image=False) for point in text_raw]
        image_results = [self._convert_point(point, is_image=True) for point in image_raw]

        logger.info(f"Retrieved {len(text_results)} text results and {len(image_results)} image results")

        return {
            "text_results": text_results,
            "image_results": image_results
        }

    def retrieve_with_images(self, k_text: int = 5, k_images: int = 3) -> Dict[str, Any]:
        """
        Retrieve results and enrich image results with base64-encoded image data.
        
        Args:
            k_text: Number of text results.
            k_images: Number of image results.
            
        Returns:
            Dictionary with text and image results (image results include base64 data).
        """
        result = self.retrieve_top_k(k_text, k_images)

        # Enforce conversion on image results (if any still remain unconverted)
        result["image_results"] = [img if isinstance(img, dict) else self._convert_point(img, is_image=True)
                                    for img in result["image_results"]]

        for i, img_result in enumerate(result["image_results"]):
            try:
                image_id = img_result.get('image_id')
                if not image_id:
                    continue

                image_data = self._get_image_from_mongodb(image_id)
                if image_data:
                    base64_str = base64.b64encode(image_data).decode('utf-8')
                    img_format = img_result.get('format', 'jpeg').lower()
                    img_result["base64_data"] = f"data:image/{img_format};base64,{base64_str}"
                    logger.info(f"Added image data for result {i+1}/{len(result['image_results'])}: {image_id}")
                else:
                    img_result["base64_data"] = f"data:image/jpeg;base64,placeholder_{image_id}"
                    logger.warning(f"Using placeholder for image {image_id}")
            except Exception as e:
                logger.error(f"Error fetching image data for {img_result.get('image_id', 'unknown')}: {str(e)}")
                img_result["base64_data"] = f"data:image/jpeg;base64,error_{img_result.get('image_id', 'unknown')}"

        return result

    def format_for_context(self, results: Dict[str, Any]) -> str:
        """
        Format retrieval results for inclusion in prompt context.
        
        Args:
            results: Dictionary with text_results and image_results.
            
        Returns:
            A formatted string.
        """
        context_parts = []

        for i, result in enumerate(results.get("text_results", [])):
            content = result.get('content', '')
            source = result.get('source', 'Unknown source')
            page_info = f" (Page {result.get('page_range')})" if result.get('page_range') else ""
            context_parts.append(f"[TEXT {i+1}] From document: {source}{page_info}\n{content}")

        for i, result in enumerate(results.get("image_results", [])):
            caption = result.get("caption", "No caption available")
            page_info = f" (Page {result.get('page_num')})" if result.get('page_num') else ""
            source_info = f" from {result.get('source')}" if result.get('source') else ""
            context_parts.append(f"[IMAGE {i+1}]{page_info}{source_info}: {caption}")

        formatted_context = "\n\n".join(context_parts)
        logger.info(f"Created formatted context with {len(results.get('text_results', []))} text chunks and {len(results.get('image_results', []))} images")
        return formatted_context
