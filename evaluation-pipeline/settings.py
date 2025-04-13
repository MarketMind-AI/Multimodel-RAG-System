from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"  # Still keep this for text
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 256
    EMBEDDING_SIZE: int = 384  # For text embeddings
    CLIP_EMBEDDING_SIZE: int = 512  # For CLIP embeddings
    EMBEDDING_MODEL_DEVICE: str = "cpu"

    # OpenAI config
    OPENAI_MODEL_ID: str = "gpt-4-1106-preview"
    OPENAI_API_KEY: str | None = None

    # QdrantDB config
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_DATABASE_URL: str = "http://localhost:6333"

    # QDRANT_CLOUD_URL: str = "str"
    # USE_QDRANT_CLOUD: bool = False
    # QDRANT_APIKEY: str | None = None

    # MQ config
    # RABBITMQ_DEFAULT_USERNAME: str = "guest"
    # RABBITMQ_DEFAULT_PASSWORD: str = "guest"
    # RABBITMQ_HOST: str = "localhost"
    # RABBITMQ_PORT: int = 5673

    # CometML config
    COMET_API_KEY: str = "Gm9kfXOp6BN3RWiTTOxYJMy3l"
    COMET_WORKSPACE: str ="mohamedayari1"
    COMET_PROJECT: str = "market-mind-inference-pipeline-prototype"

    # LLM Model config
    TOKENIZERS_PARALLELISM: str = "false"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_TYPE: str = "Meta/Llama2"

    # RAG config
    TOP_K: int = 3
    KEEP_TOP_K: int = 3
    EXPAND_N_QUERY: int = 3



    AZURE_OPENAI_API_KEY: str = "9f9e3951045f4274b63e1daf2fbcf1b0"
    AZURE_OPENAI_ENDPOINT: str = "https://market-mindaihubala0195041921.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview"
    OPENAI_API_VERSION: str = "2023-03-15-preview"



settings = AppSettings()
