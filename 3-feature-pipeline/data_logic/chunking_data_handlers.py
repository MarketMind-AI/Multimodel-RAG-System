import hashlib
from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import PdfChunkModel
from models.clean import  PdfCleanedModel
from utils.chunking import chunk_text


class ChunkingDataHandler(ABC):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @abstractmethod
    def chunk(self, data_model: DataModel) -> list[DataModel]:
        pass



class PdfChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: PdfCleanedModel) -> list[PdfChunkModel]:
        data_models_list = []

        text_content = data_model.cleaned_extracted_text
        chunks = chunk_text(text_content)

        for chunk in chunks:
            model = PdfChunkModel(                            
                entry_id= data_model.entry_id,
                source= data_model.source,
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                chunk_content=chunk,
                num_pages= data_model.num_pages,
                type= data_model.type    
            )
            data_models_list.append(model)
        return data_models_list
            
            



