import os
import base64
from pymongo import MongoClient
from datetime import datetime
import hashlib


def insert_test_pdf_to_mongodb(uri, database_name):
    """
    Insert a test PDF document into MongoDB.
    """
    client = MongoClient(uri)
    db = client[database_name]
    collection = db["pdf_documents"]

    # Create a simple test PDF document
    pdf_data = {
        "source": "test_document.pdf",
        "extracted_text": "This is a test document for multimodal RAG system.",
        "num_pages": 2,
        "upload_date": datetime.utcnow().isoformat(),
        "images": [
            {
                "image_id": "test_image_1",
                "page_num": 1,
                "size": [800, 600],
                "format": "jpeg",
                "caption": "Test image 1"
            }
        ]
    }

    try:
        result = collection.insert_one(pdf_data)
        pdf_id = result.inserted_id
        print(f"Test PDF document inserted with _id: {pdf_id}")
        return str(pdf_id)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        client.close()


def insert_test_image_to_mongodb(uri, database_name, pdf_id=None):
    """
    Insert a test image document into MongoDB.
    """
    client = MongoClient(uri)
    db = client[database_name]
    collection = db["image_documents"]

    # Create a simple test image - a 1x1 pixel red dot
    # In a real scenario, you'd use actual image data
    simple_image_data = b'\xff\x00\x00'  # Red pixel in RGB
    image_id = hashlib.md5(simple_image_data).hexdigest()

    # Create a test image document
    image_data = {
        "source": "test_document.pdf",
        "image_id": image_id,
        "image_data": simple_image_data,
        "page_num": 1,
        "size": [1, 1],
        "format": "jpeg",
        "caption": "A simple red pixel test image",
        "upload_date": datetime.utcnow().isoformat()
    }

    # Add reference to PDF if provided
    if pdf_id:
        image_data["pdf_id"] = pdf_id

    try:
        result = collection.insert_one(image_data)
        print(f"Test image document inserted with _id: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        client.close()


if __name__ == "__main__":
    # Use the connection string from your environment or hardcode for testing
    mongo_uri = "mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set"
    database_name = "production"
    
    print("Inserting test documents into MongoDB to trigger CDC...")
    
    # Insert test PDF document
    pdf_id = insert_test_pdf_to_mongodb(mongo_uri, database_name)
    
    if pdf_id:
        # Insert test image document with reference to the PDF
        insert_test_image_to_mongodb(mongo_uri, database_name, pdf_id)
    else:
        # Insert standalone test image document
        insert_test_image_to_mongodb(mongo_uri, database_name)
    
    print("Test documents inserted. Check CDC logs for processing events.")