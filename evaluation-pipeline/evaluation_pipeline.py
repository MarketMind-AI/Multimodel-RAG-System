import time

import pandas as pd
from llm.prompt_templates import InferenceTemplate
from rag.retriever import VectorRetriever
from settings import settings

from evaluation.model import evaluate_llm
from evaluation.rag import evaluate_w_ragas

from app.llama2 import llm_pipeline


from monitoring import PromptMonitoringManager


class RAGEvaluator:
    def __init__(self) -> None:
        # self.qwak_client = llm_pipeline #response = llm_pipeline(pipeline_input)
        self.template = InferenceTemplate()
        self.prompt_monitoring_manager = PromptMonitoringManager()
        self._timings = {
            "retrieval": 0.0,
            "generation": 0.0,
            "evaluation_rag": 0.0,
            "evaluation_llm": 0.0,
        }

    def generate(
        self,
        query: str,
        enable_rag: bool = True,
        enable_evaluation: bool = True,
        enable_monitoring: bool = True,
    ) -> dict:
        prompt_template = self.template.create_template(enable_rag=enable_rag)
        prompt_template_variables = {
            "question": query,
        }

        if enable_rag is True:
            st_time = time.time_ns()
            retriever = VectorRetriever(query=query)
            context = retriever.retrieve_top_k(9)  # Retrieves relevant context for the query
            prompt_template_variables["context"] = context
            prompt = prompt_template.format(question=query, context=context)
            en_time = time.time_ns()
            self._timings["retrieval"] = (en_time - st_time) / 1e9
        else:
            prompt = prompt_template.format(question=query)

        st_time = time.time_ns()
        input_ = pd.DataFrame([{"instruction": prompt}]).to_json()

        response = llm_pipeline(input_)
        
        #here you have to pay attention
        answer = response[0]["content"]
        en_time = time.time_ns()
        self._timings["generation"] = (en_time - st_time) / 1e9

        if enable_evaluation is True:
            if enable_rag:
                st_time = time.time_ns()
                rag_eval_scores = evaluate_w_ragas(
                    query=query, output=answer, context=context
                )
                en_time = time.time_ns()
                self._timings["evaluation_rag"] = (en_time - st_time) / 1e9
            st_time = time.time_ns()
            llm_eval = evaluate_llm(query=query, output=answer)
            en_time = time.time_ns()
            self._timings["evaluation_llm"] = (en_time - st_time) / 1e9
            evaluation_result = {
                "llm_evaluation": "" if not llm_eval else llm_eval,
                "rag_evaluation": {} if not rag_eval_scores else rag_eval_scores,
            }
        else:
            evaluation_result = None

        if enable_monitoring is True:
            self.prompt_monitoring_manager.log(
                prompt=prompt,
                prompt_template=prompt_template.template,
                prompt_template_variables=prompt_template_variables,
                output=answer,
            )
            self.prompt_monitoring_manager.log_chain(
                query=query,
                context=context,
                llm_gen=answer,
                llm_eval_output=evaluation_result["llm_evaluation"],
                rag_eval_scores=evaluation_result["rag_evaluation"],
                timings=self._timings,
            )

        return {"answer": answer, "llm_evaluation_result": evaluation_result}
