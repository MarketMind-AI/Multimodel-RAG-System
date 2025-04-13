from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from loguru import logger
from io import BytesIO
import os
from datetime import datetime
import json

from services.pdf_multimodal_processing import extract_content_from_pdf
from config import settings
from db.documents import PdfDocument, ImageDocument, ImageMetadata
from db.mongo import connection

# Initialize FastAPI
app = FastAPI()

@app.post("/process_pdf_file/")
async def process_pdf_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        contents = await file.read()

        # Save the uploaded file locally
        file_path = f"/tmp/{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(contents)
        logger.info(f"Saved uploaded file to {file_path}")

        # Extract text and images
        extracted_content = extract_content_from_pdf(file_path)
        extracted_text = extracted_content['text']
        extracted_images = extracted_content['images']
        num_pages = extracted_content['num_pages']
        
        logger.info(f"Extraction complete: {len(extracted_text)} chars, {len(extracted_images)} images, {num_pages} pages")
        
        # Prepare image metadata for PDF document
        image_metadata_list = []
        for img in extracted_images:
            image_metadata_list.append(
                ImageMetadata(
                    image_id=img['image_id'],
                    page_num=img['page_num'],
                    size=img['size'],
                    format=img['format']
                )
            )
        
        # Store the PDF document with image metadata
        pdf_document = PdfDocument(
            source=file.filename,
            extracted_text=extracted_text,
            num_pages=num_pages,
            upload_date=datetime.utcnow().isoformat(),
            images=image_metadata_list
        )
        
        pdf_result = pdf_document.save()
        pdf_id = str(pdf_result)
        logger.info(f"Saved PDF document with ID: {pdf_id}")
        
        # Store each image as a separate document
        image_ids = []
        for i, img in enumerate(extracted_images):
            try:
                logger.info(f"Processing image {i+1}/{len(extracted_images)}: {img['image_id']}")
                image_document = ImageDocument(
                    source=file.filename,
                    image_id=img['image_id'],
                    image_data=img['image_data'],
                    page_num=img['page_num'],
                    size=img['size'],
                    format=img['format'],
                    upload_date=datetime.utcnow().isoformat()
                )
                
                logger.info(f"Saving image {img['image_id']} to database")
                image_result = image_document.save()
                image_id = str(image_result)
                image_ids.append(image_id)
                logger.info(f"Saved image document with ID: {image_id}")
            except Exception as img_error:
                logger.error(f"Error saving image {i+1}: {str(img_error)}")
        
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {
            "status": "success", 
            "pdf_id": pdf_id,
            "extracted_text_length": len(extracted_text),
            "num_pages": num_pages,
            "image_count": len(image_ids),
            "image_ids": image_ids
        }
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {"status": "error", "message": str(e)}