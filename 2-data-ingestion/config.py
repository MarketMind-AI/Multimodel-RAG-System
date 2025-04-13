import os
from dotenv import load_dotenv, find_dotenv


# Load environment variables
load_dotenv(find_dotenv())


class Settings:
    MONGODB_PWD = os.getenv("MONGODB_PWD")
    # MONGO_DATABASE_HOST = f"mongodb+srv://zoldyck:{MONGODB_PWD}@cluster0.kjantw5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    # MongoDB configs
    # MONGO_DATABASE_HOST: str = "mongodb://localhost:30001,localhost:30002,localhost:30003/?replicaSet=my-replica-set"

    MONGO_DATABASE_HOST: str = (
        "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
    )
    MONGO_DATABASE_NAME: str = "production"


    RABBITMQ_HOST: str = "mq"  # or the Docker host if running remotely
    RABBITMQ_PORT: int = 5672  # Port mapped in Docker Compose
    RABBITMQ_DEFAULT_USERNAME: str = "guest"  # Default username
    RABBITMQ_DEFAULT_PASSWORD: str = "guest"  # Default password
    RABBITMQ_QUEUE_NAME: str = "default"


settings = Settings()
