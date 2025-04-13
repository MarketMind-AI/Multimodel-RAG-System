from typing import Optional, Tuple

from models.base import VectorDBDataModel


class PdfCleanedModel(VectorDBDataModel):
    source: str
    cleaned_extracted_text: str
    num_pages: Optional[int] 
    type: str

    
    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "source": self.source,
            "cleaned_extracted_text": self.cleaned_extracted_text,
            "num_pages": self.num_pages,
            "type": self.type,
        }

        return self.entry_id, data


