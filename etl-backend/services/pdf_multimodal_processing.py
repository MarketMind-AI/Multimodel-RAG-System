import fitz  # PyMuPDF
import os
import io
from PIL import Image
import hashlib
from loguru import logger

def extract_content_from_pdf(pdf_path: str) -> dict:
    """
    Extract both text and images from a PDF document.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        extracted_text = ""
        extracted_images = []
        
        logger.info(f"Processing PDF with {pdf_document.page_count} pages")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            
            # Extract text
            page_text = page.get_text()
            extracted_text += page_text
            logger.info(f"Extracted {len(page_text)} characters from page {page_num+1}")
            
            # Extract images
            image_list = page.get_images(full=True)
            logger.info(f"Found {len(image_list)} image references on page {page_num+1}")
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    logger.info(f"Processing image with xref {xref}")
                    
                    base_image = pdf_document.extract_image(xref)
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create unique image ID based on content hash
                    image_id = hashlib.md5(image_bytes).hexdigest()
                    logger.info(f"Generated image ID: {image_id}")
                    
                    # Convert to PIL Image for processing
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip very small images (likely icons or decorations)
                    if image.width < 100 or image.height < 100:
                        logger.info(f"Skipping small image: {image.width}x{image.height}")
                        continue
                        
                    # Create image entry
                    image_data = {
                        'image_id': image_id,
                        'page_num': page_num + 1,
                        'image_data': image_bytes,
                        'size': (image.width, image.height),
                        'format': image_ext
                    }
                    
                    extracted_images.append(image_data)
                    logger.info(f"Successfully extracted image {img_index+1}: {image.width}x{image.height}")
                except Exception as img_error:
                    logger.error(f"Error extracting image {img_index} on page {page_num+1}: {str(img_error)}")
        
        logger.info(f"Successfully extracted {len(extracted_text)} chars of text and {len(extracted_images)} images")
        return {
            'text': extracted_text,
            'images': extracted_images,
            'num_pages': pdf_document.page_count
        }
    except Exception as e:
        logger.error(f"Error extracting content from PDF: {str(e)}")
        return {
            'text': str(e),
            'images': [],
            'num_pages': 0
        }