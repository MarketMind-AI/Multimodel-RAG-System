import core.logger_utils as logger_utils
from evaluation_pipeline import RAGEvaluator

logger = logger_utils.get_logger(__name__)


if __name__ == "__main__":
    inference_endpoint = RAGEvaluator()

    query = """
        some questions about content of dropped documents 
        """

    response = inference_endpoint.generate(
        query=query,
        enable_rag=True,
        enable_evaluation=True,
        enable_monitoring=True,
    )

    logger.info(f"Answer: {response['answer']}")
    logger.info("=" * 50)
    logger.info(f"LLM Evaluation Result: {response['llm_evaluation_result']}")
