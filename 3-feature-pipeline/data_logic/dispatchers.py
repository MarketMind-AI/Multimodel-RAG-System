from datetime import datetime
from utils.logging import get_logger

from data_logic.chunking_data_handlers import (
    ChunkingDataHandler,
    PdfChunkingHandler,
)
from data_logic.cleaning_data_handlers import (
    CleaningDataHandler,
    PdfCleaningHandler,
)
from data_logic.embedding_data_handlers import (
    EmbeddingDataHandler,
    PdfEmbeddingHandler,
)

# Import image handlers
from data_logic.image_data_handlers import (
    ImageCleaningHandler,
    ImageProcessingHandler,
    ImageChunkingHandler,
    BasicImageChunkingHandler,
    ImageEmbeddingHandler,
    CLIPImageEmbeddingHandler,
)

from models.base import DataModel
from models.raw import PdfRawModel
from models.image_models import ImageRawModel, ImageCleanedModel

from typing import Union
import json

logger = get_logger(__name__)

class RawDispatcher:
    @staticmethod
    def handle_mq_message(message: dict) -> DataModel:
        data_type = message.get("type")
        logger.info("Received message.", extra={"data_type": data_type})

        if data_type == "pdf_documents":
            try:
                return PdfRawModel(
                    entry_id=message.get('entry_id'),
                    type=message.get('type'),
                    source=message.get('source'),
                    extracted_text=message.get('extracted_text'),
                    num_pages=message.get('num_pages')
                )
            except KeyError as e:
                logger.error(f"Missing key in message: {e}")
                raise ValueError(f"Invalid message format: missing {e}")
            except ValueError as e:
                logger.error(f"Value error in message: {e}")
                raise ValueError(f"Invalid message format: {e}")
        elif data_type == "image_documents":
            try:
                return ImageRawModel(
                    entry_id=message.get('entry_id'),
                    type=message.get('type'),
                    source=message.get('source'),
                    image_id=message.get('image_id'),
                    image_data=message.get('image_data'),
                    page_num=message.get('page_num'),
                    size=message.get('size'),
                    format=message.get('format'),
                    caption=message.get('caption'),
                    pdf_id=message.get('pdf_id')
                )
            except KeyError as e:
                logger.error(f"Missing key in image message: {e}")
                raise ValueError(f"Invalid image message format: missing {e}")
            except ValueError as e:
                logger.error(f"Value error in image message: {e}")
                raise ValueError(f"Invalid image message format: {e}")
        else:
            logger.error(f"Unsupported data type: {data_type}")
            raise ValueError(f"Unsupported data type: {data_type}")


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_type: str) -> Union[CleaningDataHandler, ImageCleaningHandler]:
        if data_type == "pdf_documents":
            return PdfCleaningHandler()
        elif data_type == "image_documents":
            return ImageProcessingHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


class CleaningDispatcher:
    cleaning_factory = CleaningHandlerFactory()

    @classmethod
    def dispatch_cleaner(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        clean_model = handler.clean(data_model)

        if data_type == "pdf_documents":
            logger.info(
                "PDF data cleaned successfully.",
                extra={
                    "data_type": data_type,
                    "cleaned_content_len": len(clean_model.cleaned_extracted_text),
                }
            )
        elif data_type == "image_documents":
            logger.info(
                "Image data cleaned successfully.",
                extra={
                    "data_type": data_type,
                    "image_id": clean_model.image_id,
                    "has_caption": clean_model.caption is not None,
                }
            )

        return clean_model


class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_type: str) -> Union[ChunkingDataHandler, ImageChunkingHandler]:
        if data_type == "pdf_documents":
            return PdfChunkingHandler()
        elif data_type == "image_documents":
            return BasicImageChunkingHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


class ChunkingDispatcher:
    chunking_factory = ChunkingHandlerFactory()

    @classmethod
    def dispatch_chunker(cls, data_model: DataModel) -> list[DataModel]:
        data_type = data_model.type
        handler = cls.chunking_factory.create_handler(data_type)
        chunk_models = handler.chunk(data_model)

        if data_type == "pdf_documents":
            logger.info(
                "Text content chunked successfully.",
                extra={
                    "num_chunks": len(chunk_models),
                    "data_type": data_type,
                }
            )
        elif data_type == "image_documents":
            logger.info(
                "Image chunked successfully.",
                extra={
                    "num_chunks": len(chunk_models),
                    "data_type": data_type,
                    "image_id": getattr(data_model, "image_id", "unknown"),
                }
            )

        return chunk_models


class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_type: str) -> Union[EmbeddingDataHandler, ImageEmbeddingHandler]:
        if data_type == "pdf_documents":
            return PdfEmbeddingHandler()
        elif data_type == "image_documents":
            return CLIPImageEmbeddingHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


class EmbeddingDispatcher:
    embedding_factory = EmbeddingHandlerFactory()

    @classmethod
    def dispatch_embedder(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.embedding_factory.create_handler(data_type)
        embedded_model = handler.embedd(data_model)

        if data_type == "pdf_documents":
            logger.info(
                "Text chunk embedded successfully.",
                extra={
                    "data_type": data_type,
                    "embedding_len": len(embedded_model.embedded_content),
                }
            )
        elif data_type == "image_documents":
            logger.info(
                "Image embedded successfully.",
                extra={
                    "data_type": data_type,
                    "image_id": embedded_model.image_id,
                    "embedding_shape": embedded_model.embedded_content.shape,
                }
            )

        return embedded_model
