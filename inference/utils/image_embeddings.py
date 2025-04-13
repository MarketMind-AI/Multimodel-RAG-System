"""
Utilities for generating image embeddings and captions
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logging import get_logger
from config import settings
import io
from PIL import Image
import hashlib

logger = get_logger(__name__)

# Global model instances to avoid reloading them
_text_embedding_model = None
_clip_model = None

def _get_text_model():
    """
    Get or initialize the text embedding model
    """
    global _text_embedding_model
    if _text_embedding_model is None:
        try:
            _text_embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
            logger.info(f"Initialized text embedding model: {settings.EMBEDDING_MODEL_ID}")
        except Exception as e:
            logger.error(f"Error initializing text embedding model: {str(e)}")
            raise
    
    return _text_embedding_model

def _get_clip_model():
    """
    Get or initialize the CLIP model for image embeddings
    
    In a production system, you would use a real CLIP model
    For now, we're simulating it with the text model
    """
    return _get_text_model()

def generate_text_embedding(text: str) -> np.ndarray:
    """
    Generate a text embedding using the configured model
    
    Args:
        text: The text to embed
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    try:
        model = _get_text_model()
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}")
        # Return a zero vector of the correct size as fallback
        return np.zeros(settings.EMBEDDING_SIZE)

def generate_image_caption(image_data: bytes) -> str:
    """
    Generate a caption for an image
    
    In a production system, you would use a real image captioning model
    For now, we'll generate a simple placeholder based on image features
    
    Args:
        image_data: Raw image data
        
    Returns:
        str: A caption describing the image
    """
    try:
        # Open the image to get basic properties
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        format_name = image.format or "Unknown"
        mode = image.mode
        
        # Generate a simple caption based on image properties
        caption = f"Image in {format_name} format with dimensions {width}x{height} pixels"
        
        # Add color information
        if mode == "RGB" or mode == "RGBA":
            # Sample some pixels to get color information
            image_resized = image.resize((10, 10))
            pixels = list(image_resized.getdata())
            
            # Calculate average RGB values
            avg_r = sum(p[0] for p in pixels) // len(pixels)
            avg_g = sum(p[1] for p in pixels) // len(pixels)
            avg_b = sum(p[2] for p in pixels) // len(pixels)
            
            # Determine dominant color
            max_val = max(avg_r, avg_g, avg_b)
            if avg_r == max_val and avg_r > 150:
                color = "reddish"
            elif avg_g == max_val and avg_g > 150:
                color = "greenish"
            elif avg_b == max_val and avg_b > 150:
                color = "bluish"
            elif avg_r > 200 and avg_g > 200 and avg_b > 200:
                color = "bright"
            elif avg_r < 50 and avg_g < 50 and avg_b < 50:
                color = "dark"
            else:
                color = "colored"
            
            caption += f", with predominantly {color} tones"
        
        # Generate a deterministic unique ID for the image
        image_hash = hashlib.md5(image_data).hexdigest()[:8]
        caption += f" (ID: {image_hash})"
        
        return caption
    
    except Exception as e:
        logger.error(f"Error generating image caption: {str(e)}")
        return "Image without caption"

def generate_image_embedding(caption: str = None, image_data: bytes = None, image_id: str = None) -> np.ndarray:
    """
    Generate an embedding for an image using CLIP or a similar model
    
    This function can use either the image data directly, the caption, or both.
    In a production system, you would use a real multimodal model like CLIP.
    For now, we'll simulate it using the text embedding model on the caption.
    
    Args:
        caption: Optional caption for the image
        image_data: Optional raw image data
        image_id: Optional image ID for logging
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    try:
        # If we have a caption, use that for the embedding
        if caption:
            logger.info(f"Generating text-based embedding for image {image_id or 'unknown'}")
            return generate_text_embedding(caption)
        
        # If we have image data but no caption, generate a caption first
        elif image_data:
            logger.info(f"No caption provided, generating one for image {image_id or 'unknown'}")
            generated_caption = generate_image_caption(image_data)
            return generate_text_embedding(generated_caption)
        
        # If we have neither, return a zero vector
        else:
            logger.warning(f"No caption or image data provided for image {image_id or 'unknown'}")
            return np.zeros(settings.EMBEDDING_SIZE)
    
    except Exception as e:
        logger.error(f"Error generating image embedding: {str(e)}")
        return np.zeros(settings.EMBEDDING_SIZE)

def generate_multimodal_embedding(text: str, image_data: bytes = None, image_caption: str = None) -> np.ndarray:
    """
    Generate a combined embedding for text and image
    
    In a production system, you would use a real multimodal fusion model.
    For now, we'll simulate it by combining the text and image embeddings.
    
    Args:
        text: The text to embed
        image_data: Optional raw image data
        image_caption: Optional image caption
        
    Returns:
        numpy.ndarray: The combined embedding vector
    """
    try:
        # Generate text embedding
        text_embedding = generate_text_embedding(text)
        
        # Generate image embedding
        if image_caption:
            image_embedding = generate_text_embedding(image_caption)
        elif image_data:
            image_embedding = generate_image_embedding(image_data=image_data)
        else:
            image_embedding = np.zeros_like(text_embedding)
        
        # Combine the embeddings (simple average)
        # In a real system, you might use a more sophisticated fusion method
        combined_embedding = (text_embedding + image_embedding) / 2.0
        
        # Normalize the embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding
    
    except Exception as e:
        logger.error(f"Error generating multimodal embedding: {str(e)}")
        return np.zeros(settings.EMBEDDING_SIZE)