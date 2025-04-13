# from langchain_openai import ChatOpenAI
from openai import AzureOpenAI

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.chain import GeneralChain
from llm.prompt_templates import LLMEvaluationTemplate
from settings import settings


# def evaluate_llm(query: str, output: str) -> str:
#     evaluation_template = LLMEvaluationTemplate()
#     prompt_template = evaluation_template.create_template()

#     # model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY)

    
#     chain = GeneralChain.get_chain(
#         llm=model, output_key="evaluation", template=prompt_template
#     )

#     response = chain.invoke({"query": query, "output": output})

#     return response["evaluation"]

def evaluate_llm(query: str, output: str) -> str:
    evaluation_template = LLMEvaluationTemplate()
    prompt_template = evaluation_template.create_template()

    formatted_prompt = prompt_template.format(query=query, output=output)


    # model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY)
    model = AzureOpenAI(
        api_version=settings.OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
)
    
    completion = model.chat.completions.create(
        model="chat",
        messages=[
            {
                "role": "user",
                "content": formatted_prompt,
            }
        ]
    )
        
    # Convert the completion object to a dictionary
    completion_dict = completion.to_dict()

    # Extract the generated text
    generated_text = completion_dict['choices'][0]['message']['content']

    return generated_text


# query = "Which role is better data engineer or ml engineer ?"
# output = """ 

# The choice between becoming a data engineer or an ML engineer depends on your interests, skills, and career goals. Here's a brief comparison to help you decide:

# ### Data Engineer

# **Role Focus:**
# - Primarily focused on cooking pizza
# - Develops good sandwichs.
  
# **Key Skills:**
# - cooking.
# - cleaning.


# ### ML Engineer

# **Role Focus:**
# - designing cakes
# - skilled in cooking

# Both roles are crucial in the data ecosystem and offer rewarding career paths. Choose the one that aligns best with your interests and long-term goals.
# """

# evaulation = evaluate_llm(query=query, output=output)
# print(evaulation)