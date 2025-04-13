from abc import ABC, abstractmethod
import hashlib
import io
from PIL import Image
from typing import List

from utils.logging import get_logger
from models.base import DataModel
from models.image_models import (
    ImageRawModel,
    ImageCleanedModel,
    ImageChunkModel,
    ImageEmbeddedModel
)
from utils.image_embeddings import generate_image_embedding, generate_image_caption

logger = get_logger(__name__)

class ImageCleaningHandler(ABC):
    """
    Abstract class for image cleaning handlers.
    Handles operations like basic image metadata extraction
    """
    
    @abstractmethod
    def clean(self, data_model: DataModel) -> DataModel:
        pass

class ImageProcessingHandler(ImageCleaningHandler):
    """
    Handler for processing raw images
    """
    
    def clean(self, data_model: ImageRawModel) -> ImageCleanedModel:
        """
        Process the raw image data:
        - Basic image validation
        - Generate caption if missing
        - Extract metadata
        """
        try:
            # Open image to verify it's valid
            if hasattr(data_model, 'image_data') and data_model.image_data:
                image = Image.open(io.BytesIO(data_model.image_data)).convert('RGB')
                width, height = image.size
                
                # Generate a caption if one doesn't exist
                caption = data_model.caption
                if not caption:
                    caption = generate_image_caption(data_model.image_data)
                    logger.info(f"Generated caption for image {data_model.image_id}: {caption}")
                
                # Create and return the cleaned model
                return ImageCleanedModel(
                    entry_id=data_model.entry_id,
                    source=data_model.source,
                    image_id=data_model.image_id,
                    page_num=data_model.page_num,
                    size=(width, height),  # Use actual size from image
                    format=data_model.format,
                    caption=caption,
                    pdf_id=data_model.pdf_id,
                    type=data_model.type
                )
            else:
                # If no image data, use original dimensions and generate a placeholder caption
                logger.warning(f"No image data found for {data_model.image_id}, using metadata only")
                return ImageCleanedModel(
                    entry_id=data_model.entry_id,
                    source=data_model.source,
                    image_id=data_model.image_id,
                    page_num=data_model.page_num,
                    size=data_model.size if hasattr(data_model, 'size') else (0, 0),
                    format=data_model.format if hasattr(data_model, 'format') else "unknown",
                    caption=data_model.caption if hasattr(data_model, 'caption') and data_model.caption else "No image data available",
                    pdf_id=data_model.pdf_id if hasattr(data_model, 'pdf_id') else None,
                    type=data_model.type
                )
            
        except Exception as e:
            logger.error(f"Error cleaning image {data_model.image_id if hasattr(data_model, 'image_id') else 'unknown'}: {str(e)}")
            # Return original data with basic cleaning
            return ImageCleanedModel(
                entry_id=data_model.entry_id,
                source=data_model.source,
                image_id=data_model.image_id if hasattr(data_model, 'image_id') else "unknown",
                page_num=data_model.page_num if hasattr(data_model, 'page_num') else 0,
                size=data_model.size if hasattr(data_model, 'size') else (0, 0),
                format=data_model.format if hasattr(data_model, 'format') else "unknown",
                caption=data_model.caption if hasattr(data_model, 'caption') else "Error processing image",
                pdf_id=data_model.pdf_id if hasattr(data_model, 'pdf_id') else None,
                type=data_model.type
            )


class ImageChunkingHandler(ABC):
    """
    Abstract class for image chunking handlers.
    Handles breaking down images into regions if needed.
    """
    
    @abstractmethod
    def chunk(self, data_model: DataModel) -> List[DataModel]:
        pass


class BasicImageChunkingHandler(ImageChunkingHandler):
    """
    Basic implementation that treats the entire image as a single chunk
    For more advanced implementations, this could perform object detection
    or image segmentation to extract meaningful regions
    """
    
    def chunk(self, data_model: ImageCleanedModel) -> List[ImageChunkModel]:
        """
        Create a single chunk for the whole image
        """
        # Create a chunk ID based on image ID
        chunk_id = hashlib.md5(f"{data_model.image_id}_full".encode()).hexdigest()
        
        # Create a chunk covering the entire image
        width, height = data_model.size
        
        chunk = ImageChunkModel(
            entry_id=data_model.entry_id,
            source=data_model.source,
            image_id=data_model.image_id,
            chunk_id=chunk_id,
            region=(0, 0, width, height),  # Full image region
            caption=data_model.caption,
            pdf_id=data_model.pdf_id,
            type=data_model.type
        )
        
        logger.info(f"Created single chunk for image {data_model.image_id}")
        return [chunk]


class ImageEmbeddingHandler(ABC):
    """
    Abstract class for image embedding handlers.
    Handles generating vector embeddings for images.
    """
    
    @abstractmethod
    def embedd(self, data_model: DataModel) -> DataModel:
        pass


class CLIPImageEmbeddingHandler(ImageEmbeddingHandler):
    """
    Handler for generating image embeddings using CLIP or similar models
    """
    
    def embedd(self, data_model: ImageChunkModel) -> ImageEmbeddedModel:
        """
        Generate an embedding for the image
        """
        try:
            # Generate embedding using the utility function
            # This can use either the caption, the image, or both
            embedding = generate_image_embedding(
                caption=data_model.caption,
                image_id=data_model.image_id
            )
            
            logger.info(f"Generated embedding for image {data_model.image_id} with shape {embedding.shape}")
            
            # Get size and format from the metadata
            # For a chunk, we need to reconstruct some fields
            size = data_model.region[2:] if hasattr(data_model, 'region') else (0, 0)
            
            return ImageEmbeddedModel(
                entry_id=data_model.entry_id,
                source=data_model.source,
                image_id=data_model.image_id,
                page_num=data_model.page_num if hasattr(data_model, 'page_num') else 0,
                size=size,
                format="unknown",  # Format might not be available in chunks
                caption=data_model.caption,
                pdf_id=data_model.pdf_id if hasattr(data_model, 'pdf_id') else None,
                embedded_content=embedding,
                type=data_model.type
            )
            
        except Exception as e:
            logger.error(f"Error generating embedding for image {data_model.image_id}: {str(e)}")
            # Generate a fallback embedding
            import numpy as np
            from config import settings
            
            # Create a zero vector of the appropriate size
            fallback_embedding = np.zeros(settings.EMBEDDING_SIZE)
            
            logger.warning(f"Using fallback zero embedding for image {data_model.image_id}")
            
            return ImageEmbeddedModel(
                entry_id=data_model.entry_id,
                source=data_model.source,
                image_id=data_model.image_id,
                page_num=data_model.page_num if hasattr(data_model, 'page_num') else 0,
                size=(0, 0),  # Default size when error occurs
                format="unknown",
                caption=data_model.caption,
                pdf_id=data_model.pdf_id if hasattr(data_model, 'pdf_id') else None,
                embedded_content=fallback_embedding,
                type=data_model.type
            )