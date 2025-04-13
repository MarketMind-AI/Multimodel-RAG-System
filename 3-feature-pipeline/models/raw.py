from datetime import datetime
from typing import Optional, Tuple

from models.base import DataModel, VectorDBDataModel





class PdfRawModel(DataModel):
    source: str
    extracted_text: str
    num_pages: Optional[int] 









