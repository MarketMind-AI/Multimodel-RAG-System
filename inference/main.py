from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from inference_pipeline import RAG_Orchestrator
from utils.logging import get_logger
import requests

logger = get_logger(__name__)
orchestrator = RAG_Orchestrator()

LLM_SERVICE_URL = "http://host.docker.internal:5005"
MULTIMODAL_ENDPOINT = "/generate_multimodal"
TEXT_ENDPOINT = "/generate_answer"
IMAGE_ENDPOINT = "/generate_with_image"

app = FastAPI(
    title="Multimodal RAG API",
    description="API for multimodal retrieval-augmented generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this as needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    """Request model for the query endpoint"""
    query: str
    image_query: Optional[str] = None
    k_text: Optional[int] = 5
    k_images: Optional[int] = 3
    include_image_data: Optional[bool] = False
    generate_answer: Optional[bool] = False
    use_llava: Optional[bool] = True

class QueryResponse(BaseModel):
    """Response model for the query endpoint"""
    prompt: str
    context: str
    text_results: List[Dict[str, Any]]
    image_results: List[Dict[str, Any]]
    answer: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "multimodal": True}

def process_scored_point(point: Any) -> Dict[str, Any]:
    """
    Convert a scored point (dictionary or object) to a standardized dictionary.
    This ensures that dictionary methods (like .get()) can be used downstream.
    """
    if isinstance(point, dict):
        return point
    else:
        payload = getattr(point, "payload", {}) or {}
        return {
            "id": getattr(point, "id", ""),
            "score": getattr(point, "score", 0.0),
            "payload": payload
        }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a multimodal query.
    
    This endpoint takes a text query and an optional image query,
    retrieves relevant content, and returns it with a generated prompt.
    If generate_answer is True, it will also call the LLM service to generate a response.
    """
    try:
        if request.k_text < 1:
            raise HTTPException(status_code=400, detail="k_text must be >= 1")
        if request.k_images < 0:
            raise HTTPException(status_code=400, detail="k_images must be >= 0")
        
        logger.info(f"Processing query: '{request.query}' with image query: '{request.image_query}'")
        logger.info(f"Parameters: k_text={request.k_text}, k_images={request.k_images}, include_image_data={request.include_image_data}")
        
        if request.include_image_data:
            if request.use_llava and request.k_images > 0:
                result = orchestrator.prepare_for_llava(
                    query=request.query, 
                    image_query=request.image_query,
                    k_text=request.k_text, 
                    k_images=request.k_images
                )
                response = {
                    "prompt": result["prompt"],
                    "context": result.get("text_context", ""),
                    "text_results": [process_scored_point(point) for point in result["all_results"]["text_results"]],
                    "image_results": [process_scored_point(point) for point in result["all_results"]["image_results"]]
                }
            else:
                raw_response = orchestrator.retrieve_with_images(
                    query=request.query, 
                    image_query=request.image_query,
                    k_text=request.k_text, 
                    k_images=request.k_images
                )
                response = {
                    "prompt": raw_response["prompt"],
                    "context": raw_response["context"],
                    "text_results": [process_scored_point(point) for point in raw_response["text_results"]],
                    "image_results": [process_scored_point(point) for point in raw_response["image_results"]]
                }
        else:
            raw_response = orchestrator.retrieve(
                query=request.query, 
                image_query=request.image_query,
                k_text=request.k_text, 
                k_images=request.k_images
            )
            response = {
                "prompt": raw_response["prompt"],
                "context": raw_response["context"],
                "text_results": [process_scored_point(point) for point in raw_response["text_results"]],
                "image_results": [process_scored_point(point) for point in raw_response["image_results"]]
            }
        
        logger.info(f"Retrieved {len(response['text_results'])} text results and {len(response['image_results'])} image results")
        
        if request.generate_answer:
            try:
                # Ensure image results are in dictionary form.
                image_results = [process_scored_point(img) for img in response.get("image_results", [])]
                has_images = request.include_image_data and any("base64_data" in img for img in image_results)
                
                if has_images and request.use_llava:
                    if "best_image" in result and result["best_image"] and "base64_data" in result["best_image"]:
                        llm_payload = {
                            "prompt": request.query,
                            "image_data": result["best_image"]["base64_data"],
                            "context": response["context"]
                        }
                        llm_response = requests.post(
                            f"{LLM_SERVICE_URL}{IMAGE_ENDPOINT}",
                            json=llm_payload,
                            timeout=60
                        )
                    else:
                        llm_payload = {
                            "prompt": request.query,
                            "context": response["context"]
                        }
                        llm_response = requests.post(
                            f"{LLM_SERVICE_URL}{TEXT_ENDPOINT}",
                            json=llm_payload,
                            timeout=30
                        )
                else:
                    llm_payload = {
                        "prompt": request.query,
                        "context": response["context"]
                    }
                    llm_response = requests.post(
                        f"{LLM_SERVICE_URL}{TEXT_ENDPOINT}",
                        json=llm_payload,
                        timeout=30
                    )
                
                if llm_response.status_code == 200:
                    llm_result = llm_response.json()
                    response["answer"] = llm_result.get("answer", "No response generated.")
                    logger.info("Generated answer using LLM service")
                else:
                    logger.error(f"Error from LLM service: {llm_response.status_code} - {llm_response.text}")
            except Exception as e:
                logger.error(f"Error calling LLM service: {str(e)}")
                response["answer"] = f"Error generating answer: {str(e)}"
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process_query")
async def legacy_process_query(request: Request):
    """Legacy endpoint for backward compatibility"""
    try:
        data = await request.json()
        query_request = QueryRequest(
            query=data.get("query", ""),
            image_query=data.get("image_query"),
            k_text=data.get("k_text", 5),
            k_images=data.get("k_images", 3),
            include_image_data=data.get("include_image_data", False),
            generate_answer=data.get("generate_answer", False),
            use_llava=data.get("use_llava", True)
        )
        return await process_query(query_request)
    
    except Exception as e:
        logger.error(f"Error in legacy endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
