class Settings:
    # URL for the inference service inside Docker
    INFERENCE_SERVICE_URL = "http://inference:8000/query"
    
    # URLs for LLM services
    TEXT_LLM_SERVICE_URL = "http://host.docker.internal:5005/generate_answer"
    MULTIMODAL_LLM_SERVICE_URL = "http://host.docker.internal:5005/generate_multimodal"
    
    # URL for image processing service
    IMAGE_LLM_SERVICE_URL = "http://host.docker.internal:5005/generate_with_image"


settings = Settings()