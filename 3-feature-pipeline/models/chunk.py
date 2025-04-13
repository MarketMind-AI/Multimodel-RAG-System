from datetime import datetime
from typing import Optional

from models.base import DataModel

class PdfChunkModel(DataModel):
    entry_id: str
    source: str
    chunk_id: str
    chunk_content: str
    num_pages: Optional[int] 
    type: str

