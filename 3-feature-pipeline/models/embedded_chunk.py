from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from models.base import VectorDBDataModel



class PdfEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    source: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    num_pages: Optional[int] 
    type: str

    
    class Config:
        arbitrary_types_allowed = True

    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.entry_id,
            "source": self.source,
            "content": self.chunk_content,
            "num_pages": self.num_pages,
            "type": self.type,
        }

        return self.chunk_id, self.embedded_content, data
