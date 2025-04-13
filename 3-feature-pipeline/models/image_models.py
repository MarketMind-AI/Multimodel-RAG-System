from typing import Optional, Tuple, List
import numpy as np

from models.base import DataModel, VectorDBDataModel


class ImageRawModel(DataModel):
    """
    Raw image model from MongoDB documents collection
    """
    entry_id: str
    source: str
    image_id: str
    image_data: bytes
    page_num: int
    size: Tuple[int, int]
    format: str
    caption: Optional[str] = None
    pdf_id: Optional[str] = None
    type: str = "image_documents"


class ImageCleanedModel(VectorDBDataModel):
    """
    Cleaned image model with metadata
    """
    entry_id: str
    source: str
    image_id: str
    page_num: int
    size: Tuple[int, int]
    format: str
    caption: Optional[str] = None
    pdf_id: Optional[str] = None
    type: str = "image_documents"
    
    def to_payload(self) -> Tuple[str, dict]:
        """Convert to payload for non-vector storage"""
        data = {
            "source": self.source,
            "image_id": self.image_id,
            "page_num": self.page_num,
            "size": self.size,
            "format": self.format,
            "caption": self.caption,
            "pdf_id": self.pdf_id,
            "type": self.type,
        }

        return self.entry_id, data


class ImageChunkModel(DataModel):
    """
    Image chunk model - for segmentation/region extraction if needed
    This could be used for processing different regions of complex images
    """
    entry_id: str
    source: str
    image_id: str
    chunk_id: str  # Unique ID for this image chunk/region
    region: Tuple[int, int, int, int]  # (x1, y1, x2, y2) coordinates
    caption: Optional[str] = None
    pdf_id: Optional[str] = None
    type: str = "image_documents"


class ImageEmbeddedModel(VectorDBDataModel):
    """
    Image with embedded vector representation
    """
    entry_id: str
    source: str
    image_id: str
    page_num: int
    size: Tuple[int, int]
    format: str
    caption: Optional[str] = None
    pdf_id: Optional[str] = None
    embedded_content: np.ndarray
    type: str = "image_documents"
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        """Convert to payload for vector storage"""
        data = {
            "id": self.entry_id,
            "source": self.source,
            "image_id": self.image_id,
            "page_num": self.page_num,
            "size": self.size,
            "format": self.format,
            "caption": self.caption,
            "pdf_id": self.pdf_id,
            "type": self.type,
        }

        return self.image_id, self.embedded_content, data