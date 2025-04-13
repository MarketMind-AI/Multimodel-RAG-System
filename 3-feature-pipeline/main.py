import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import bytewax.operators as op
from bytewax.dataflow import Dataflow

from db import QdrantDatabaseConnector

from data_flow.stream_input import RabbitMQSource
from data_flow.stream_output import QdrantOutput
from data_logic.dispatchers import (
    ChunkingDispatcher,
    CleaningDispatcher,
    EmbeddingDispatcher,
    RawDispatcher,
)

from utils.logging import get_logger

logger = get_logger(__name__)

def create_multimodal_pipeline():
    """
    Create a dataflow pipeline that handles both text and image processing
    """
    # Initialize Qdrant connection
    connection = QdrantDatabaseConnector()
    
    # Create a new dataflow
    flow = Dataflow("Multimodal ingestion pipeline")
    
    # Input stream from RabbitMQ
    stream = op.input("input", flow, RabbitMQSource())
    
    # Parse message and create appropriate data model
    stream = op.map("raw dispatch", stream, RawDispatcher.handle_mq_message)
    
    # Clean data (text extraction, image processing)
    stream = op.map("clean dispatch", stream, CleaningDispatcher.dispatch_cleaner)
    
    # Output cleaned data to Qdrant
    op.output(
        "cleaned data insert to qdrant",
        stream,
        QdrantOutput(connection=connection, sink_type="clean"),
    )
    
    # Chunk data (text paragraphs, image regions)
    stream = op.flat_map("chunk dispatch", stream, ChunkingDispatcher.dispatch_chunker)
    
    # Generate embeddings for chunks
    stream = op.map(
        "embedded chunk dispatch", stream, EmbeddingDispatcher.dispatch_embedder
    )
    
    # Output embedded data to Qdrant vector store
    op.output(
        "embedded data insert to qdrant",
        stream,
        QdrantOutput(connection=connection, sink_type="vector"),
    )
    
    logger.info("Multimodal ingestion pipeline created successfully")
    return flow

# Create the flow for Bytewax
flow = create_multimodal_pipeline()