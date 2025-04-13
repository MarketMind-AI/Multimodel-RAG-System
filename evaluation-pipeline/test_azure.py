from openai import AzureOpenAI
from settings import settings

from llm.chain import GeneralChain
from llm.prompt_templates import LLMEvaluationTemplate


import llm.prompt_templates as templates

from datasets import Dataset
from langchain_openai import ChatOpenAI

from llm.chain import GeneralChain
from pandas import DataFrame

from ragas import evaluate
from ragas.embeddings import HuggingfaceEmbeddings



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
    harmfulness,
]

model = AzureOpenAI(
        api_version=settings.OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY
)

embd_model = HuggingfaceEmbeddings(model=settings.EMBEDDING_MODEL_ID)


result = evaluate(
    amnesty_qa["eval"],
    metrics=metrics, 
    llm=model,
    embeddings=embd_model
)

print(result)


# def evaluate_llm(query: str, output: str) -> str:
#     evaluation_template = LLMEvaluationTemplate()
#     prompt_template = evaluation_template.create_template()

#     formatted_prompt = prompt_template.format(query=query, output=output)


#     # model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=settings.OPENAI_API_KEY)
#     model = AzureOpenAI(
#         api_version=settings.OPENAI_API_VERSION,
#         azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
#         api_key=settings.AZURE_OPENAI_API_KEY
# )
    
#     completion = model.chat.completions.create(
#         model="chat",
#         messages=[
#             {
#                 "role": "user",
#                 "content": formatted_prompt,
#             }
#         ]
#     )
        
#     # Convert the completion object to a dictionary
#     completion_dict = completion.to_dict()

#     # Extract the generated text
#     generated_text = completion_dict['choices'][0]['message']['content']

#     return generated_text


query = "Which role is better data engineer or ml engineer ?"
output = """ 

The choice between becoming a data engineer or an ML engineer depends on your interests, skills, and career goals. Here's a brief comparison to help you decide:

### Data Engineer

**Role Focus:**
- Primarily focused on building and maintaining the infrastructure and architecture for data generation, storage, and processing.
- Develops data pipelines and manages databases.
  
**Key Skills:**
- Proficiency in big data technologies (e.g., Hadoop, Spark).
- Strong knowledge of SQL and database management systems.
- Programming skills, often in Python, Java, or Scala.
- Experience with cloud platforms (e.g., AWS, Azure, GCP).

**Career Path:**
- Typically involved in data architecture design and optimization.
- Suited for those interested in data management and system engineering.

### ML Engineer

**Role Focus:**
- Focuses on designing and deploying machine learning models.
- Works closely with data scientists to implement algorithms into production.

**Key Skills:**
- Strong understanding of machine learning algorithms and frameworks (e.g., TensorFlow, PyTorch).
- Programming skills, typically in Python or R.
- Ability to tune models and ensure scalability.
- Knowledge of data transformation and feature engineering.

**Career Path:**
- Involves more applied mathematics and statistics.
- Ideal for those interested in model development and AI applications.

### Considerations:

- **Interest in Systems vs. Algorithms:** If you prefer systems and infrastructure, data engineering might be a better fit. If you like working with algorithms and models, consider ML engineering.

- **Current Skill Set:** Assess your current skills and see which role aligns more closely with your strengths. 

- **Career Goals:** Consider where you see yourself in the future. Both roles offer growth opportunities but may lead to different specializations (e.g., data architect vs. AI specialist).

Both roles are crucial in the data ecosystem and offer rewarding career paths. Choose the one that aligns best with your interests and long-term goals.
"""
context = """ 
Data Engineer
Role Focus:

Specializes in creating and managing the infrastructure and systems for data generation, storage, and processing.
Designs and maintains data pipelines and databases.
Key Skills:

Expertise in big data technologies such as Hadoop and Spark.
Strong proficiency in SQL and database management.
Programming experience, often in Python, Java, or Scala.
Familiarity with cloud platforms like AWS, Azure, or GCP.
Career Path:

Typically involves designing and optimizing data architectures.
Ideal for those with an interest in data systems and engineering.
ML Engineer
Role Focus:

Concentrates on developing and deploying machine learning models.
Collaborates with data scientists to put algorithms into production.
Key Skills:

Deep knowledge of machine learning algorithms and frameworks like TensorFlow and PyTorch.
Programming skills, usually in Python or R.
Experience in tuning models and ensuring their scalability.
Understanding of data transformation and feature engineering.
Career Path:

Focuses on applied mathematics and statistics.
Best suited for those interested in model development and AI technologies.

"""


# evaulation = evaluate_llm(query=query, output=output)
# print(evaulation)

# client = AzureOpenAI(
#     api_version=settings.OPENAI_API_VERSION,
#     azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
#     api_key=settings.AZURE_OPENAI_API_KEY
# )


# completion = client.chat.completions.create(
#     model="chat",
#     messages=[
#         {
#             "role": "user",
#             "content": "Which role is better data engineer or ml engineer ?",
#         }
#     ]
# )
      
# # Convert the completion object to a dictionary
# completion_dict = completion.to_dict()

# # Extract the generated text
# generated_text = completion_dict['choices'][0]['message']['content']

# # Print the extracted text
# print((generated_text))



