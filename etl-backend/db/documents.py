import uuid
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, UUID4, ConfigDict
from errors import ImproperlyConfigured
from pymongo import errors
from utils import get_logger
from datetime import datetime
from db.mongo import connection

_database = connection.get_database("production")
logger = get_logger(__name__)

class BaseDocument(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    
    def from_mongo(cls, data: dict):
        if not data:
            return data

        id = data.pop("_id", None)
        return cls(**dict(data, id=id))
    
    def to_mongo(self, **kwargs) -> dict:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.dict(
            exclude_unset=exclude_unset, by_alias=by_alias, **kwargs
        )

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        return parsed
    
    def save(self, **kwargs):
        collection = _database[self._get_collection_name()]

        try:
            result = collection.insert_one(self.to_mongo(**kwargs))
            return result.inserted_id
        except errors.WriteError:
            logger.exception("Failed to insert document.")
            return None
    
    @classmethod
    def bulk_insert(cls, documents: List, **kwargs) -> Optional[List[str]]:
        collection = _database[cls._get_collection_name()]
        try:
            result = collection.insert_many(
                [doc.to_mongo(**kwargs) for doc in documents]
            )
            return result.inserted_ids
        except errors.WriteError:
            logger.exception("Failed to insert documents.")
            return None
    
    @classmethod
    def get_or_create(cls, **filter_options) -> Optional[str]:
        collection = _database[cls._get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return str(cls.from_mongo(instance).id)
            new_instance = cls(**filter_options)
            new_instance = new_instance.save()
            return new_instance
        except errors.OperationFailure:
            logger.exception("Failed to retrieve or create document.")
            return None
    
    @classmethod
    def _get_collection_name(cls):
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define a Settings configuration class with the name of the collection."
            )
        return cls.Settings.name

class ImageMetadata(BaseModel):
    """Metadata for an image extracted from a document"""
    image_id: str
    page_num: int
    size: tuple
    format: str
    caption: Optional[str] = None

class PdfDocument(BaseDocument):
    source: str
    extracted_text: str
    num_pages: Optional[int] = None
    upload_date: str
    images: List[ImageMetadata] = []
    
    class Settings:
        name = "pdf_documents"
        
class ImageDocument(BaseDocument):
    source: str
    image_id: str
    image_data: bytes  # Binary image data
    page_num: int
    size: tuple
    format: str
    caption: Optional[str] = None
    upload_date: str
    
    class Settings:
        name = "image_documents"
    
    def to_mongo(self, **kwargs) -> dict:
        """Override to_mongo to handle binary data properly"""
        # Get base dictionary from parent method
        doc_dict = super().to_mongo(**kwargs)
        
        # Ensure binary data is properly stored
        if isinstance(doc_dict.get('image_data'), bytes):
            # MongoDB can store binary data directly
            # But you might want to add additional logging
            logger.info(f"Preparing binary image data for storage: {len(doc_dict['image_data'])} bytes")
        
        return doc_dict
    
    class Settings:
        name = "image_documents"