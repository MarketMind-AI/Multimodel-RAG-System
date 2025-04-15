import pandas as pd
from llm.prompt_templates import InferenceTemplate
from rag.multimodal_retriever import MultimodalRetriever
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

class RAG_Orchestrator:
    def __init__(self) -> None:
        self.template = InferenceTemplate()
        logger.info("Initialized RAG_Orchestrator with multimodal retrieval capabilities")

    def retrieve(self, query: str, image_query: str = None, enable_rag: bool = True,
                k_text: int = 5, k_images: int = 3, marketing_options: dict = None) -> dict:
        """
        Retrieve relevant content based on the query and generate a prompt
        
        Args:
            query: Text query
            image_query: Optional image-specific query
            enable_rag: Whether to use RAG
            k_text: Number of text results to retrieve
            k_images: Number of image results to retrieve
            marketing_options: Dict of marketing-specific options
            
        Returns:
            Dictionary with prompt, context, text_results, and image_results
        """
        # Determine if this is a multimodal query (image_query provided or k_images > 0)
        is_multimodal = (image_query is not None) or (k_images > 0)
        
        # Check if marketing content generation is enabled
        is_marketing = marketing_options is not None and marketing_options.get('enabled', False)
        
        # Select appropriate prompt template
        prompt_template = self.template.create_template(
            enable_rag=enable_rag, 
            multimodal=is_multimodal,
            marketing=is_marketing,
            platform=marketing_options.get('platform') if is_marketing else None,
            target_audience=marketing_options.get('target_audience') if is_marketing else None,
            tone=marketing_options.get('tone') if is_marketing else None,
            format=marketing_options.get('format') if is_marketing else None,
            content_length=marketing_options.get('content_length') if is_marketing else None
        )
        
        # Initialize prompt variables
        prompt_template_variables = {"question": query}

        if enable_rag:
            # Use the multimodal retriever
            retriever = MultimodalRetriever(query=query, image_query=image_query)
            retrieval_results = retriever.retrieve_top_k(k_text=k_text, k_images=k_images)
            
            # Format retrieval results for context
            context = retriever.format_for_context(retrieval_results)
            
            # Add context to prompt variables
            prompt_template_variables["context"] = context
            
            # Add marketing variables if applicable
            if is_marketing:
                prompt_template_variables["platform"] = marketing_options.get('platform', 'social media')
                prompt_template_variables["target_audience"] = marketing_options.get('target_audience', 'general audience')
                prompt_template_variables["tone"] = marketing_options.get('tone', 'professional')
                prompt_template_variables["format"] = marketing_options.get('format', 'social media post')
                prompt_template_variables["content_length"] = marketing_options.get('content_length', 'medium')
                
                logger.info(f"Generated marketing prompt for {marketing_options.get('platform')} with {marketing_options.get('tone')} tone")
            
            # Generate the final prompt
            prompt = prompt_template.format(**prompt_template_variables)
            
            logger.info(f"Generated {'marketing' if is_marketing else 'standard'} prompt with context from {len(retrieval_results['text_results'])} text chunks and {len(retrieval_results['image_results'])} images")
            
            return {
                "prompt": prompt,
                "context": context,
                "text_results": retrieval_results["text_results"],
                "image_results": retrieval_results["image_results"],
                "marketing_options": marketing_options if is_marketing else None
            }
        else:
            # If RAG is disabled, just return the prompt without context
            if is_marketing:
                prompt_template_variables.update({
                    "platform": marketing_options.get('platform', 'social media'),
                    "target_audience": marketing_options.get('target_audience', 'general audience'),
                    "tone": marketing_options.get('tone', 'professional'),
                    "format": marketing_options.get('format', 'social media post'),
                    "content_length": marketing_options.get('content_length', 'medium'),
                    "context": "No context provided."
                })
            
            prompt = prompt_template.format(**prompt_template_variables)
            logger.info("Generated prompt without RAG context")
            
            return {
                "prompt": prompt,
                "context": "",
                "text_results": [],
                "image_results": [],
                "marketing_options": marketing_options if is_marketing else None
            }
    
    def retrieve_with_images(self, query: str, image_query: str = None,
                          k_text: int = 5, k_images: int = 3,
                          marketing_options: dict = None) -> dict:
        """
        Retrieve content including actual image data for frontend display
        
        This method is similar to retrieve() but includes base64-encoded
        image data for display in a frontend and use with LLaVA.
        
        Args:
            query: Main text query
            image_query: Optional image-specific query
            k_text: Number of text results to retrieve
            k_images: Number of image results to retrieve
            marketing_options: Dict of marketing-specific options
            
        Returns:
            Dictionary with prompt, context, text_results, and image_results (with base64 data)
        """
        # First get the basic retrieval results without images
        result = self.retrieve(query, image_query, True, k_text, k_images, marketing_options)
        
        # If we have image results and need to include the binary data
        if result["image_results"]:
            # Get the retriever with images
            retriever = MultimodalRetriever(query=query, image_query=image_query)
            retrieval_results = retriever.retrieve_with_images(k_text=k_text, k_images=k_images)
            
            # Update image_results with the data that includes base64 images
            result["image_results"] = retrieval_results["image_results"]
            
            logger.info(f"Added image data to {len(result['image_results'])} image results")
        
        return result
        
    def prepare_for_llava(self, query: str, image_query: str = None,
                          k_text: int = 5, k_images: int = 1,
                          marketing_options: dict = None) -> dict:
        """
        Prepare data specifically for LLaVA model
        
        This method optimizes for LLaVA by prioritizing the highest-scoring image
        and preparing a prompt that works well with LLaVA's capabilities.
        
        Args:
            query: The main query
            image_query: Optional image-focused query
            k_text: Number of text results to retrieve
            k_images: Number of images to retrieve (typically just 1 for LLaVA)
            marketing_options: Dict of marketing-specific options
            
        Returns:
            Dictionary with data prepared for LLaVA
        """
        # Get results with images included
        results = self.retrieve_with_images(query, image_query, k_text, k_images, marketing_options)
        
        # Extract the highest-scoring image if available
        best_image = None
        if results["image_results"]:
            # Sort by score and take the best one
            sorted_images = sorted(results["image_results"], key=lambda x: x.get("score", 0), reverse=True)
            best_image = sorted_images[0] if sorted_images else None
        
        # Prepare text context by formatting text results
        text_context = ""
        if results["text_results"]:
            text_parts = []
            for i, result in enumerate(results["text_results"]):
                text_parts.append(f"Document {i+1}: {result['content']}")
            text_context = "\n\n".join(text_parts)
        
        # Create an optimized prompt for LLaVA
        llava_prompt = f"Question: {query}\n\n"
        
        if text_context:
            llava_prompt += f"Additional context:\n{text_context}\n\n"
        
        # Add marketing specific information if applicable
        if marketing_options and marketing_options.get('enabled', False):
            llava_prompt += f"\nTarget audience: {marketing_options.get('target_audience', 'general audience')}\n"
            llava_prompt += f"Platform: {marketing_options.get('platform', 'social media')}\n"
            llava_prompt += f"Tone: {marketing_options.get('tone', 'professional')}\n"
            llava_prompt += f"Format: {marketing_options.get('format', 'social media post')}\n"
            llava_prompt += f"Content length: {marketing_options.get('content_length', 'medium')}\n"
            
        # Create result dictionary
        llava_data = {
            "prompt": llava_prompt,
            "query": query,
            "text_context": text_context,
            "best_image": best_image,
            "marketing_options": marketing_options,
            "all_results": results
        }
        
        logger.info(f"Prepared {'marketing' if marketing_options and marketing_options.get('enabled') else 'standard'} data for LLaVA with {len(results['text_results'])} text chunks and {1 if best_image else 0} primary image")
        
        return llava_data