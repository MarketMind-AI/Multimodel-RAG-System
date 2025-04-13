from langchain_community.llms import Ollama
import base64
from io import BytesIO
from PIL import Image
import requests
from typing import Optional, Dict, Any, List

# Load Ollama LLaVA model
llava_model = Ollama(model="llava", request_timeout=120)  # 120 seconds timeout


class LLaVAProcessor:
    """
    LLaVA multimodal processor for handling text and image inputs
    """
    
    def __init__(self):
        self.model = llava_model
    
    def generate_from_text(self, prompt: str, context: str) -> str:
        """Generate a response using only text input"""
        complete_prompt = f"""
        You are a knowledgeable assistant. Given the following context, answer the question.
        Context: {context}
        
        Question: {prompt}
        
        Answer:"""
        
        return self.model(complete_prompt)
    
    def generate_with_image(self, prompt: str, image_data: str, context: Optional[str] = None) -> str:
        """
        Generate a response using both text and image input
        
        Args:
            prompt: Text prompt/question
            image_data: Base64 encoded image data (with MIME type prefix)
            context: Optional additional text context
        
        Returns:
            Generated response
        """
        # Extract base64 data from the data URL
        if "base64," in image_data:
            # Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
            base64_str = image_data.split("base64,")[1]
        else:
            base64_str = image_data
            
        complete_prompt = f"""
        Examine the image and answer the following question.
        
        Question: {prompt}
        """
        
        if context:
            complete_prompt += f"""
            
            Additional context: {context}
            """
            
        # LLaVA requires the base64 data for image input
        # Format according to Ollama's multimodal input format
        response = self.model.invoke(complete_prompt, images=[base64_str])
        
        return response
    
    def process_multimodal_query(self, 
                               prompt: str,
                               text_results: List[Dict[str, Any]] = None, 
                               image_results: List[Dict[str, Any]] = None,
                               context: str = None) -> str:
        """
        Process a multimodal query with retrieved results
        
        Args:
            prompt: The user's question
            text_results: Retrieved text results
            image_results: Retrieved image results with base64 data
            context: Formatted context string
            
        Returns:
            Generated response
        """
        # If we have image results with base64 data
        if image_results and any("base64_data" in img for img in image_results):
            # Get the highest scoring image
            sorted_images = sorted(
                [img for img in image_results if "base64_data" in img], 
                key=lambda x: x.get("score", 0), 
                reverse=True
            )
            
            if sorted_images:
                # Use the first/best image
                best_image = sorted_images[0]
                image_data = best_image["base64_data"]
                
                # Generate response using the image
                return self.generate_with_image(prompt, image_data, context)
        
        # If no usable images, fall back to text-only generation
        return self.generate_from_text(prompt, context or "")