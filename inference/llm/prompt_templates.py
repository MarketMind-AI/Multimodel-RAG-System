from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from pydantic import BaseModel


class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self, *args) -> PromptTemplate:
        pass


class InferenceTemplate(BasePromptTemplate):
    """
    Template provider for different inference scenarios:
    - Simple: No RAG
    - RAG: Text-only RAG
    - Multimodal RAG: Text and images
    """
    
    # Simple prompt without any RAG context
    simple_prompt: str = """You are an AI language model assistant. Your task is to generate a cohesive and concise response to the user question.
    
    Question: {question}
    """

    # Basic RAG prompt for text-only retrieval
    rag_prompt: str = """You are a specialist in technical content writing. Your task is to create technical content based on a user query given a specific context 
    with additional information consisting of relevant passages from documents.
    
    Here is a list of steps that you need to follow in order to solve this task:
    Step 1: You need to analyze the user provided query: {question}
    Step 2: You need to analyze the provided context and how the information in it relates to the user question: {context}
    Step 3: Generate the content keeping in mind that it needs to be as cohesive and concise as possible related to the subject presented in the query and similar to the users writing style and knowledge presented in the context.
    """
    
    # Enhanced multimodal RAG prompt for text and image retrieval
    multimodal_rag_prompt: str = """You are a specialist in technical content writing with expertise in multimodal information analysis. Your task is to create comprehensive technical content based on a user query, given specific context with both text passages and image descriptions.
    
    Here is a list of steps that you need to follow:
    
    Step 1: Analyze the user provided query: {question}
    
    Step 2: Carefully analyze the provided context, which includes both text content and image descriptions:
    {context}
    
    Note that:
    - Text content is marked with [TEXT #] and provides written information from documents
    - Image content is marked with [IMAGE #] and provides captions or descriptions of images
    
    Step 3: Generate comprehensive content that:
    - Integrates information from both textual passages and images
    - Explicitly references relevant images when they support your explanation (e.g., "As shown in Image 1...")
    - Maintains a technical and informative tone
    - Focuses on answering the query with specific details from the provided context
    - Avoids making up information not present in the context
    
    If the context contains technical diagrams or visual representations, make sure to explain their relevance to the query.
    """
    
    # LLaVA-specific prompt for image analysis
    llava_image_prompt: str = """Analyze this image and answer the following question:
    
    Question: {question}
    
    Provide a detailed analysis focusing on the visual elements that are relevant to the question.
    """
    
    # LLaVA-specific prompt for multimodal analysis with additional context
    llava_context_prompt: str = """Analyze this image in relation to the following question and context.
    
    Question: {question}
    
    Additional context:
    {context}
    
    Provide a detailed analysis that integrates both the visual information and the provided context.
    """

    def create_template(self, enable_rag: bool = True, multimodal: bool = False, use_llava: bool = False) -> PromptTemplate:
        """
        Create the appropriate prompt template
        
        Args:
            enable_rag: Whether to use RAG context
            multimodal: Whether the context includes images
            use_llava: Whether to use the LLaVA-specific prompt for image analysis
            
        Returns:
            The appropriate PromptTemplate
        """
        if use_llava:
            if enable_rag:
                # LLaVA with additional context
                return PromptTemplate(
                    template=self.llava_context_prompt,
                    input_variables=["question", "context"]
                )
            else:
                # LLaVA with just the image
                return PromptTemplate(
                    template=self.llava_image_prompt,
                    input_variables=["question"]
                )
        elif enable_rag:
            if multimodal:
                # Multimodal RAG prompt for text and images
                return PromptTemplate(
                    template=self.multimodal_rag_prompt, 
                    input_variables=["question", "context"]
                )
            else:
                # Text-only RAG prompt
                return PromptTemplate(
                    template=self.rag_prompt, 
                    input_variables=["question", "context"]
                )
        else:
            # Simple prompt without RAG
            return PromptTemplate(
                template=self.simple_prompt, 
                input_variables=["question"]
            )