# import bytewax.operators as op
# from bytewax.dataflow import Dataflow

# from db import QdrantDatabaseConnector

# from data_flow.stream_input import RabbitMQSource
# from data_flow.stream_output import QdrantOutput
# from data_logic.dispatchers import (
#     ChunkingDispatcher,
#     CleaningDispatcher,
#     EmbeddingDispatcher,
#     RawDispatcher,
# )

# connection = QdrantDatabaseConnector()

# # Define a dummy output function to just print the received messages
# def print_message(message):
#     print(message)
#     return message  # Returning the message unchanged for now

# connection = QdrantDatabaseConnector()

# flow = Dataflow("Streaming ingestion pipeline")
# stream = op.input("input", flow, RabbitMQSource())
# stream = op.map("raw dispatch", stream, RawDispatcher.handle_mq_message)

# # Output the received message for verification
# op.map("print message", stream, print_message)

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

# Initialize Qdrant connection
connection = QdrantDatabaseConnector()

# Create a new dataflow
flow = Dataflow("Streaming ingestion pipeline")

# Read from RabbitMQ
stream = op.input("input", flow, RabbitMQSource())

# Process the data
stream = op.map("raw dispatch", stream, RawDispatcher.handle_mq_message)
stream = op.map("clean dispatch", stream, CleaningDispatcher.dispatch_cleaner)


op.output(
    "cleaned data insert to qdrant",
    stream,
    QdrantOutput(connection=connection, sink_type="clean"),
)

stream = op.flat_map("chunk dispatch", stream, ChunkingDispatcher.dispatch_chunker)
stream = op.map(
    "embedded chunk dispatch", stream, EmbeddingDispatcher.dispatch_embedder
)
op.output(
    "embedded data insert to qdrant",
    stream,
    QdrantOutput(connection=connection, sink_type="vector"),
)