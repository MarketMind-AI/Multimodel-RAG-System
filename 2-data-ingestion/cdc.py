import json
import logging
import base64

from bson import json_util
from mq import publish_to_rabbitmq

from config import settings
from db import MongoDatabaseConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def stream_process():
    try:
        # Setup MongoDB connection
        client = MongoDatabaseConnector()
        db = client["production"] 
        logging.info("Connected to MongoDB.")

        # Watch changes in specific collections
        # We're watching both pdf_documents and image_documents
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert"]},
                    "ns.coll": {"$in": ["pdf_documents", "image_documents"]}
                }
            }
        ]
        
        changes = db.watch(pipeline)
        for change in changes:
            collection_name = change["ns"]["coll"]
            entry_id = str(change["fullDocument"]["_id"])  # Convert ObjectId to string
            
            # Create a copy of the document to modify
            document = dict(change["fullDocument"])
            document.pop("_id")
            document["type"] = collection_name
            document["entry_id"] = entry_id
            
            # Special handling for image data (if present)
            if collection_name == "image_documents" and "image_data" in document:
                try:
                    # For binary data, we need to encode as base64 for JSON serialization
                    binary_data = document["image_data"]
                    
                    # Remove binary data from the document
                    document.pop("image_data")
                    
                    # Add base64 encoded data
                    if binary_data:
                        # Convert BSON Binary to Python bytes if needed
                        if hasattr(binary_data, "decode"):
                            binary_data = binary_data
                            
                        # Encode as base64 and convert to string
                        document["image_data_base64"] = base64.b64encode(binary_data).decode('utf-8')
                        logging.info(f"Encoded binary image data to base64, size: {len(document['image_data_base64'])} chars")
                    else:
                        document["image_data_base64"] = ""
                        logging.warning("Empty image data detected in document")
                except Exception as e:
                    logging.error(f"Error processing image data: {str(e)}")
                    document["image_data_base64"] = ""  # Empty string as fallback
            
            # Use json_util to serialize the document
            data = json.dumps(document, default=json_util.default)
            logging.info(f"Change detected in {collection_name} and serialized: {entry_id}")
            
            # Send data to rabbitmq
            publish_to_rabbitmq(queue_name=settings.RABBITMQ_QUEUE_NAME, data=data)
            logging.info(f"Data published to RabbitMQ queue: {settings.RABBITMQ_QUEUE_NAME}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    stream_process()