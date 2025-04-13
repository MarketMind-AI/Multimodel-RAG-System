class AppSettings:
    # Embeddings config
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 256
    EMBEDDING_SIZE: int = 384  # For text embeddings
    EMBEDDING_MODEL_DEVICE: str = "cpu"
    
    # CLIP config
    CLIP_MODEL: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    CLIP_EMBEDDING_SIZE: int = 512  # For CLIP embeddings

    # QdrantDB config
    QDRANT_DATABASE_PORT: int = 6333
    USE_QDRANT_CLOUD: bool = False # if True, fill in QDRANT_CLOUD_URL and QDRANT_APIKEY
    QDRANT_CLOUD_URL: str | None = None
    QDRANT_APIKEY: str | None = None
    QDRANT_DATABASE_HOST: str = "qdrant" # or localhost if running outside Docker
    # QDRANT_DATABASE_HOST="localhost"

    # RAG config
    TOP_K: int = 5
    KEEP_TOP_K: int = 5

    # LLM Model config
    TOKENIZERS_PARALLELISM: str = "false"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_TYPE: str = "mistralai/Mistral-7B-Instruct-v0.1"


settings = AppSettings()