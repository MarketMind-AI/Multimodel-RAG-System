import io
import torch
import open_clip
from PIL import Image
import numpy as np
from typing import Union, Optional
from utils.logging import get_logger

logger = get_logger(__name__)

# Global model cache
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None

def _get_clip_model():
    """Initialize and cache the CLIP model"""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    
    if _clip_model is None:
        try:
            # Load a medium-sized model that balances performance and resource usage
            model_name = "ViT-B-32"
            pretrained = "laion2b_s34b_b79k"
            
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(model_name)
            
            # Move to CPU or GPU as available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            
            # Cache the model components
            _clip_model = model
            _clip_preprocess = preprocess
            _clip_tokenizer = tokenizer
            
            logger.info(f"Initialized OpenCLIP model: {model_name} ({pretrained}) on {device}")
        except Exception as e:
            logger.error(f"Error initializing OpenCLIP model: {str(e)}")
            raise
    
    return _clip_model, _clip_preprocess, _clip_tokenizer

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
        device = next(model.parameters()).device
        
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
        model, preprocess, _ = _get_clip_model()
        device = next(model.parameters()).device
        
        # Load and preprocess the image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # Convert to numpy and return
        return image_features.cpu().numpy()[0]
    except Exception as e:
        logger.error(f"Error generating CLIP image embedding: {str(e)}")
        return np.zeros(512)  # CLIP typically uses 512-dim embeddings
        
def generate_multimodal_embedding(
    text: Optional[str] = None, 
    image_data: Optional[bytes] = None,
    weight_text: float = 0.5
) -> np.ndarray:
    """
    Generate a combined embedding from text and image
    
    Args:
        text: Optional text to embed
        image_data: Optional image bytes to embed
        weight_text: Weight for text embedding (0-1)
        
    Returns:
        numpy.ndarray: The combined embedding vector
    """
    if text and image_data:
        # Get both embeddings
        text_embedding = generate_clip_text_embedding(text)
        image_embedding = generate_clip_image_embedding(image_data)
        
        # Weighted combination
        combined = weight_text * text_embedding + (1 - weight_text) * image_embedding
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
            
        return combined
    elif text:
        return generate_clip_text_embedding(text)
    elif image_data:
        return generate_clip_image_embedding(image_data)
    else:
        logger.warning("No text or image provided for multimodal embedding")
        return np.zeros(512)