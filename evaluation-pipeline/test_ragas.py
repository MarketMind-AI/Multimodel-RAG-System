from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)
from pandas import DataFrame
from ragas import evaluate

from ragas.metrics.critique import harmfulness
from openai import AzureOpenAI
from settings import settings
from ragas.embeddings import HuggingfaceEmbeddings
from datasets import Dataset

# list of metrics we're going to use
# metrics = [
#     faithfulness,
#     answer_relevancy,
#     context_recall,
#     context_precision,
#     harmfulness,
# ]

model = AzureOpenAI(
        api_version=settings.OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
)


embd_model = HuggingfaceEmbeddings(model=settings.EMBEDDING_MODEL_ID)




def evaluate_w_ragas(query: str, context: list[str], output: str) -> DataFrame:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "retrieved_contexts": [context],  # Context as a list of strings
        "ground_truth": ["".join(context)],  # Ground Truth as a concatenated string of all contexts
    }


    dataset = Dataset.from_dict(data_sample)

    return dataset




# data
from datasets import load_dataset
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)
from ragas.metrics.critique import harmfulness

# list of metrics we're going to use
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    # harmfulness,
]
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
print(amnesty_qa)

# result = evaluate(
#     amnesty_qa["eval"],
#     metrics=metrics, 
#     llm=model, 
#     embeddings=embd_model
# )



