from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.llama2 import llm_pipeline

from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Set up CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the specific origin of your frontend in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Expose an endpoint that will receive a prompt and context from the context retriever
@app.post("/generate_answer")
async def generate_answer(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    context = data.get("context")

    if not prompt or not context:
        return {"error": "Prompt and context are required."}

    # Design ChatPrompt Template to include both prompt and context
    template = """
    You are a knowledgeable assistant. Given the following context, answer the question.
    Context: {context}
    
    Question: {prompt}
    
    Answer:"""

    generated_prompt = ChatPromptTemplate.from_template(template)
    
    # Prepare the input for the pipeline using keyword arguments directly
    pipeline_input = generated_prompt.format_prompt(context=context, prompt=prompt).to_string()
    
    # Generate the answer using the LLM pipeline
    response = llm_pipeline(pipeline_input)
    
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5005)

