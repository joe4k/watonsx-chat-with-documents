from llama_index.core import ServiceContext

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

def eval(llm,embed_model,query,response):
    # Define service context for watsonx for evaluation
    service_context_watsonx = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    
    # Define Faithfulness and Relevancy Evaluators which are based on watsonx
    faithfulness_watsonx = FaithfulnessEvaluator(service_context=service_context_watsonx)
    relevancy_watsonx = RelevancyEvaluator(service_context=service_context_watsonx)

    faithfulness_result = faithfulness_watsonx.evaluate_response(response=response).passing
        
    relevancy_result = relevancy_watsonx.evaluate_response(query=query, response=response).passing

    return faithfulness_result,relevancy_result

