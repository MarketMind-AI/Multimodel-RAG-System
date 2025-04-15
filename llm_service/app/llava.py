from langchain_community.llms import Ollama
import base64
from io import BytesIO
from PIL import Image
import requests
from typing import Optional, Dict, Any, List

# Load Ollama LLaVA model
llava_model = Ollama(model="llava")  # 120 seconds timeout


class LLaVAProcessor:
    """
    LLaVA multimodal processor for handling text and image inputs
    """
    
    def __init__(self):
        self.model = llava_model
    
    def generate_from_text(self, prompt: str, context: str, marketing: bool = False, 
                          platform: str = None, target_audience: str = None,
                          tone: str = None, format: str = None, content_length: str = None) -> str:
        """Generate a response using only text input"""
        
        if marketing and platform and target_audience and tone and format:
            complete_prompt = f"""
            You are a professional marketing content creator for small businesses. 
            Your task is to create compelling marketing content based on the user's query and provided knowledge.
            
            REQUEST: {prompt}
            
            KNOWLEDGE BASE:
            {context}
            
            MARKETING PARAMETERS:
            - Target Platform: {platform}
            - Target Audience: {target_audience}
            - Tone: {tone}
            - Format: {format}
            - Content Length: {content_length}
            
            Create high-quality marketing content that:
            1. Has an attention-grabbing headline
            2. Uses persuasive language that resonates with the target audience
            3. Maintains the specified tone and voice for the platform
            4. Incorporates key details from the knowledge base
            5. Includes relevant calls-to-action
            6. Is optimized for the specified platform and length
            7. Follows best practices for marketing content
            
            MARKETING CONTENT:
            """
        else:
            complete_prompt = f"""
            You are a knowledgeable assistant. Given the following context, answer the question.
            Context: {context}
            
            Question: {prompt}
            
            Answer:"""
        
        return self.model(complete_prompt)
    
    def generate_with_image(self, prompt: str, image_data: str, context: Optional[str] = None,
                           marketing: bool = False, platform: str = None, 
                           target_audience: str = None, tone: str = None, 
                           format: str = None, content_length: str = None) -> str:
        """
        Generate a response using both text and image input
        
        Args:
            prompt: Text prompt/question
            image_data: Base64 encoded image data (with MIME type prefix)
            context: Optional additional text context
            marketing: Whether this is a marketing generation request
            platform, target_audience, tone, format, content_length: Marketing parameters
        
        Returns:
            Generated response
        """
        # Extract base64 data from the data URL
        if "base64," in image_data:
            # Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
            base64_str = image_data.split("base64,")[1]
        else:
            base64_str = image_data
        
        if marketing and platform and target_audience and tone and format:
            complete_prompt = f"""
            You are a professional marketing content creator for small businesses with expertise in visual content.
            Your task is to create compelling marketing content based on the following request and image.
            
            REQUEST: {prompt}
            
            KNOWLEDGE BASE:
            {context if context else "No additional knowledge provided."}
            
            MARKETING PARAMETERS:
            - Target Platform: {platform}
            - Target Audience: {target_audience}
            - Tone: {tone}
            - Format: {format}
            - Content Length: {content_length}
            
            Study the image carefully. Create high-quality marketing content that:
            1. Has an attention-grabbing headline
            2. References visual elements from the image where relevant
            3. Uses persuasive language that resonates with the target audience
            4. Maintains the specified tone and voice for the platform
            5. Incorporates key details from the knowledge base and image
            6. Includes relevant calls-to-action
            7. Is optimized for the specified platform and length
            8. Follows best practices for visual marketing content
            
            MARKETING CONTENT:
            """
        else:
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
                               context: str = None,
                               marketing: bool = False,
                               platform: str = None,
                               target_audience: str = None, 
                               tone: str = None,
                               format: str = None,
                               content_length: str = None) -> dict:
        """
        Process a multimodal query with retrieved results
        
        Args:
            prompt: The user's question
            text_results: Retrieved text results
            image_results: Retrieved image results with base64 data
            context: Formatted context string
            marketing: Whether this is a marketing request
            platform, target_audience, tone, format, content_length: Marketing parameters
            
        Returns:
            Dictionary with answer and optional formatted content
        """
        marketing_params = {
            "marketing": marketing,
            "platform": platform,
            "target_audience": target_audience,
            "tone": tone,
            "format": format,
            "content_length": content_length
        }
        
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
                response_text = self.generate_with_image(
                    prompt, 
                    image_data, 
                    context,
                    **{k: v for k, v in marketing_params.items() if v is not None}
                )
                
                if marketing:
                    # Return both the raw response and structured marketing content
                    return {
                        "answer": response_text,
                        "formatted_content": {
                            "platform": platform,
                            "target_audience": target_audience,
                            "tone": tone,
                            "format": format,
                            "content_length": content_length,
                            "content": response_text
                        }
                    }
                else:
                    return {"answer": response_text}
        
        # If no usable images, fall back to text-only generation
        response_text = self.generate_from_text(
            prompt, 
            context or "", 
            **{k: v for k, v in marketing_params.items() if v is not None}
        )
        
        if marketing:
            # Return both the raw response and structured marketing content
            return {
                "answer": response_text,
                "formatted_content": {
                    "platform": platform,
                    "target_audience": target_audience,
                    "tone": tone,
                    "format": format,
                    "content_length": content_length,
                    "content": response_text
                }
            }
        else:
            return {"answer": response_text}