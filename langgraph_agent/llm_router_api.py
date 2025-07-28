from fastapi import FastAPI, Request
from router_llm import workflow

app = FastAPI()

@app.post("/route")
async def route_prompt(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    result = workflow.invoke({"prompt": prompt})
    return {"route_result": result["result"]}
