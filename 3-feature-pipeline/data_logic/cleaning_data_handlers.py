from abc import ABC, abstractmethod

from models.base import DataModel
from models.clean import PdfCleanedModel
from models.raw import PdfRawModel
from utils.cleaning import clean_text


class CleaningDataHandler(ABC):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DataModel) -> DataModel:
        pass


class PdfCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PdfRawModel) -> PdfCleanedModel:
        return PdfCleanedModel(
            entry_id= data_model.entry_id,
            source= data_model.source,
            cleaned_extracted_text= clean_text("".join(data_model.extracted_text)),
            num_pages= data_model.num_pages,
            type= data_model.type,
        )

