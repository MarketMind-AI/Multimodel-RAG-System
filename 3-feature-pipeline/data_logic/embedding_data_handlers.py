from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import PdfChunkModel
from models.embedded_chunk import PdfEmbeddedChunkModel

from utils.embeddings import embedd_text


class EmbeddingDataHandler(ABC):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    @abstractmethod
    def embedd(self, data_model: DataModel) -> DataModel:
        pass


class PdfEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: PdfChunkModel) -> PdfEmbeddedChunkModel:
        return PdfEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            source=data_model.source,
            chunk_id=data_model.chunk_id,
            chunk_content=data_model.chunk_content,
            embedded_content=embedd_text(data_model.chunk_content),
            num_pages=data_model.num_pages,
            type=data_model.type,
        )

