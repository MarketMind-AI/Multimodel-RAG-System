"""
Utilities for generating image embeddings and captions using OpenCLIP
"""
import io
import numpy as np
import hashlib
from PIL import Image
from typing import Optional
import torch
from utils.logging import get_logger
from config import settings
logger = get_logger(__name__)

# Global model instances to avoid reloading them
_text_embedding_model = None
_clip_model = None
_clip_processor = None
_clip_tokenizer = None
_device = None

def _get_device():
    """Get the best available device for running models"""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device}")
    return _device

def _get_text_model():
    """
    Get or initialize the text embedding model
    """
    global _text_embedding_model
    if _text_embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _text_embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
            logger.info(f"Initialized text embedding model: {settings.EMBEDDING_MODEL_ID}")
        except Exception as e:
            logger.error(f"Error initializing text embedding model: {str(e)}")
            raise
    
    return _text_embedding_model

def _get_clip_model():
    """
    Get or initialize the CLIP model for image-text embeddings
    """
    global _clip_model, _clip_processor, _clip_tokenizer
    
    if _clip_model is None:
        try:
            import open_clip
            
            # Choose model size based on your requirements and resources
            # Smaller model: "ViT-B/32"
            # Larger model: "ViT-L/14"
            model_name = "ViT-B-32"
            pretrained = "laion2b_s34b_b79k"
            
            device = _get_device()
            
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(model_name)
            
            model = model.to(device)
            model.eval()
            
            _clip_model = model
            _clip_processor = preprocess
            _clip_tokenizer = tokenizer
            
            logger.info(f"Initialized CLIP model: {model_name} ({pretrained})")
        except ImportError:
            logger.error("Failed to import open_clip. Please install with: pip install open-clip-torch")
            raise
        except Exception as e:
            logger.error(f"Error initializing CLIP model: {str(e)}")
            raise
    
    return _clip_model, _clip_processor, _clip_tokenizer

def generate_text_embedding(text: str) -> np.ndarray:
    """
    Generate a text embedding using the configured text model
    
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

def generate_clip_text_embedding(text: str) -> np.ndarray:
    """
    Generate a text embedding using OpenCLIP
    
    Args:
        text: The text to embed
        
    Returns:
        numpy.ndarray: The normalized embedding vector
    """
    try:
        model, _, tokenizer = _get_clip_model()
        device = _get_device()
        
        # Tokenize and encode
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy and return
        return text_features.cpu().numpy()[0]
    except Exception as e:
        logger.error(f"Error generating CLIP text embedding: {str(e)}")
        return np.zeros(512)  # CLIP typically uses 512-dim embeddings

def generate_clip_image_embedding(image_data: bytes) -> np.ndarray:
    """
    Generate an image embedding using OpenCLIP
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        numpy.ndarray: The normalized embedding vector
    """
    try:
        model, processor, _ = _get_clip_model()
        device = _get_device()
        
        # Load and preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_input = processor(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # Convert to numpy and return
        return image_features.cpu().numpy()[0]
    except Exception as e:
        logger.error(f"Error generating CLIP image embedding: {str(e)}")
        return np.zeros(512)  # CLIP typically uses 512-dim embeddings

def generate_image_caption(image_data: bytes) -> str:
    """
    Generate a caption for an image
    
    In a production system, you would use a proper image captioning model like BLIP
    For now, we generate a simple placeholder based on image features
    
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
    Generate an embedding for an image
    
    This function can use either the image data directly, the caption, or both.
    With CLIP integration, it will use CLIP for image embedding when possible.
    
    Args:
        caption: Optional caption for the image
        image_data: Optional raw image data
        image_id: Optional image ID for logging
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    try:
        # If we have image data, use CLIP directly on the image
        if image_data:
            logger.info(f"Generating CLIP image embedding for image {image_id or 'unknown'}")
            return generate_clip_image_embedding(image_data)
        
        # If we have a caption but no image, use CLIP text embedding
        elif caption:
            logger.info(f"Generating CLIP text embedding for image {image_id or 'unknown'} based on caption")
            return generate_clip_text_embedding(caption)
        
        # If we have neither, return a zero vector
        else:
            logger.warning(f"No caption or image data provided for image {image_id or 'unknown'}")
            return np.zeros(512)  # CLIP uses 512-dim vectors
    
    except Exception as e:
        logger.error(f"Error generating image embedding: {str(e)}")
        return np.zeros(512)  # CLIP uses 512-dim vectors

def generate_multimodal_embedding(text: str, image_data: bytes = None, image_caption: str = None) -> np.ndarray:
    """
    Generate a combined embedding for text and image using CLIP
    
    When text and image are both available, this creates a combined embedding.
    
    Args:
        text: The text to embed
        image_data: Optional raw image data
        image_caption: Optional image caption
        
    Returns:
        numpy.ndarray: The combined embedding vector
    """
    try:
        # If we have image data, use it directly with CLIP
        if image_data:
            # Get text and image embeddings from CLIP
            text_embedding = generate_clip_text_embedding(text)
            image_embedding = generate_clip_image_embedding(image_data)
            
            # Simple combination (average)
            combined_embedding = (text_embedding + image_embedding) / 2.0
            
            # Normalize
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm
            
            return combined_embedding
        
        # If we have an image caption but no image data
        elif image_caption:
            # Get text embeddings for the query and the image caption
            query_embedding = generate_clip_text_embedding(text)
            caption_embedding = generate_clip_text_embedding(image_caption)
            
            # Simple combination (average)
            combined_embedding = (query_embedding + caption_embedding) / 2.0
            
            # Normalize
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm
            
            return combined_embedding
        
        # If we only have text, just return the text embedding
        else:
            return generate_clip_text_embedding(text)
    
    except Exception as e:
        logger.error(f"Error generating multimodal embedding: {str(e)}")
        return np.zeros(512)  # CLIP uses 512-dim vectors