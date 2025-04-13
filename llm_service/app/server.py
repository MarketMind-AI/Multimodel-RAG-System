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

class MultimodalRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    text_results: Optional[List[Dict[str, Any]]] = None
    image_results: Optional[List[Dict[str, Any]]] = None

# Endpoint for text-only generation (backward compatibility)
@app.post("/generate_answer")
async def generate_answer(request: Request):
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

# New endpoint for multimodal generation
@app.post("/generate_multimodal")
async def generate_multimodal(request: MultimodalRequest):
    try:
        # Process the multimodal query
        response = llava_processor.process_multimodal_query(
            prompt=request.prompt,
            text_results=request.text_results,
            image_results=request.image_results,
            context=request.context
        )
        
        return {"answer": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Alternative endpoint using a single image
@app.post("/generate_with_image")
async def generate_with_image(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        image_data = data.get("image_data")  # Base64 encoded image
        context = data.get("context")
        
        if not prompt or not image_data:
            return {"error": "Prompt and image_data are required."}
        
        # Generate response with image
        response = llava_processor.generate_with_image(prompt, image_data, context)
        
        return {"answer": response}
    
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON data provided."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5005, timeout_keep_alive=120)  # 120 seconds timeout
