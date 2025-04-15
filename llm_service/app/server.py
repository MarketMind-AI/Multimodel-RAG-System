from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import sys
import os

# Add parent directory to Python path to resolve import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import LLaVA processor - handle different import patterns
try:
    from app.llava import LLaVAProcessor
except ModuleNotFoundError:
    try:
        from llava import LLaVAProcessor  # Try local import if we're in the app directory
    except ModuleNotFoundError:
        raise ImportError("Could not import LLaVAProcessor. Please check your project structure.")

app = FastAPI(
    title="Multimodal LLM Service",
    description="Service for generating responses using LLaVA multimodal model",
    version="1.0.0"
)

# Set up CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the specific origin of your frontend in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLaVA processor
llava_processor = LLaVAProcessor()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Define the request models for better validation
class TextRequest(BaseModel):
    prompt: str
    context: str
    marketing: Optional[bool] = False
    platform: Optional[str] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    format: Optional[str] = None
    content_length: Optional[str] = None

class MultimodalRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    text_results: Optional[List[Dict[str, Any]]] = None
    image_results: Optional[List[Dict[str, Any]]] = None
    marketing: Optional[bool] = False
    platform: Optional[str] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    format: Optional[str] = None
    content_length: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str
    image_data: str
    context: Optional[str] = None
    marketing: Optional[bool] = False
    platform: Optional[str] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    format: Optional[str] = None
    content_length: Optional[str] = None

# Endpoint for text-only generation
@app.post("/generate_answer")
async def generate_answer(request: TextRequest):
    try:
        # Generate text-only response with optional marketing parameters
        if request.marketing:
            response = llava_processor.generate_from_text(
                prompt=request.prompt,
                context=request.context,
                marketing=True,
                platform=request.platform,
                target_audience=request.target_audience,
                tone=request.tone,
                format=request.format,
                content_length=request.content_length
            )
            
            # Structure the response for marketing content
            return {
                "answer": response,
                "formatted_content": {
                    "platform": request.platform,
                    "target_audience": request.target_audience,
                    "tone": request.tone,
                    "format": request.format,
                    "content_length": request.content_length,
                    "content": response
                }
            }
        else:
            # Standard response
            response = llava_processor.generate_from_text(request.prompt, request.context)
            return {"answer": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Legacy endpoint for backward compatibility with text-only requests
@app.post("/legacy_generate_answer")
async def legacy_generate_answer(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        context = data.get("context")

        if not prompt:
            return {"error": "Prompt is required."}

        # Generate text-only response
        response = llava_processor.generate_from_text(prompt, context or "")
        
        return {"answer": response}
    
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON data provided. Please send a proper JSON object with 'prompt' and 'context' fields."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Endpoint for multimodal generation
@app.post("/generate_multimodal")
async def generate_multimodal(request: MultimodalRequest):
    try:
        # Process the multimodal query with optional marketing parameters
        response_data = llava_processor.process_multimodal_query(
            prompt=request.prompt,
            text_results=request.text_results,
            image_results=request.image_results,
            context=request.context,
            marketing=request.marketing,
            platform=request.platform,
            target_audience=request.target_audience,
            tone=request.tone,
            format=request.format,
            content_length=request.content_length
        )
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Alternative endpoint using a single image
@app.post("/generate_with_image")
async def generate_with_image(request: ImageRequest):
    try:
        # Generate response with image and optional marketing parameters
        if request.marketing:
            response = llava_processor.generate_with_image(
                prompt=request.prompt,
                image_data=request.image_data,
                context=request.context,
                marketing=True,
                platform=request.platform,
                target_audience=request.target_audience,
                tone=request.tone,
                format=request.format,
                content_length=request.content_length
            )
            
            # Structure the response for marketing content
            return {
                "answer": response,
                "formatted_content": {
                    "platform": request.platform,
                    "target_audience": request.target_audience,
                    "tone": request.tone,
                    "format": request.format,
                    "content_length": request.content_length,
                    "content": response
                }
            }
        else:
            # Standard response with image
            response = llava_processor.generate_with_image(
                request.prompt, 
                request.image_data, 
                request.context
            )
            return {"answer": response}
    
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON data provided."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Marketing-specific endpoints
@app.post("/generate_marketing")
async def generate_marketing(request: TextRequest):
    """
    Specialized endpoint for marketing content generation with text only
    """
    try:
        # Ensure marketing parameters are set
        if not request.platform or not request.target_audience or not request.tone or not request.format:
            raise HTTPException(
                status_code=400,
                detail="Marketing content generation requires platform, target_audience, tone, and format parameters."
            )
        
        # Generate marketing content
        response = llava_processor.generate_from_text(
            prompt=request.prompt,
            context=request.context,
            marketing=True,
            platform=request.platform,
            target_audience=request.target_audience,
            tone=request.tone,
            format=request.format,
            content_length=request.content_length
        )
        
        # Return structured response
        return {
            "answer": response,
            "formatted_content": {
                "platform": request.platform,
                "target_audience": request.target_audience,
                "tone": request.tone,
                "format": request.format,
                "content_length": request.content_length,
                "content": response
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/generate_marketing_with_image")
async def generate_marketing_with_image(request: ImageRequest):
    """
    Specialized endpoint for marketing content generation with image
    """
    try:
        # Ensure marketing parameters are set
        if not request.platform or not request.target_audience or not request.tone or not request.format:
            raise HTTPException(
                status_code=400,
                detail="Marketing content generation requires platform, target_audience, tone, and format parameters."
            )
        
        # Generate marketing content with image
        response = llava_processor.generate_with_image(
            prompt=request.prompt,
            image_data=request.image_data,
            context=request.context,
            marketing=True,
            platform=request.platform,
            target_audience=request.target_audience,
            tone=request.tone,
            format=request.format,
            content_length=request.content_length
        )
        
        # Return structured response
        return {
            "answer": response,
            "formatted_content": {
                "platform": request.platform,
                "target_audience": request.target_audience,
                "tone": request.tone,
                "format": request.format,
                "content_length": request.content_length,
                "content": response
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5005, timeout_keep_alive=200)  # 120 seconds timeout