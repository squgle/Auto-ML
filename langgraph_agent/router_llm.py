import os
import google.generativeai as genai
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool
from typing_extensions import TypedDict

# === Configure Gemini API ===
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# ✅ Define the correct schema with `input` key (not `prompt`)
class State(TypedDict):
    input: str

# === LangChain tools ===
preprocessing_tool = Tool(
    name="PreprocessingTool",
    func=lambda x: "Starting preprocessing pipeline...",
    description="Cleans and prepares raw data."
)

ml_modeling_tool = Tool(
    name="MLModelingTool",
    func=lambda x: "Starting ML modeling pipeline...",
    description="Trains machine learning models."
)

reporting_tool = Tool(
    name="ReportingTool",
    func=lambda x: "Starting reporting pipeline...",
    description="Generates business and analytical reports."
)

default_tool = Tool(
    name="DefaultTool",
    func=lambda x: "Sorry, I couldn't understand the task.",
    description="Handles unknown tasks."
)

# === Build LangGraph ===
graph = StateGraph(state_schema=State)

# === Add ToolNode-compatible LangChain tools ===
graph.add_node("preprocessing", ToolNode([preprocessing_tool]))
graph.add_node("ml_modeling", ToolNode([ml_modeling_tool]))
graph.add_node("reporting", ToolNode([reporting_tool]))
graph.add_node("default", ToolNode([default_tool]))

# === Router function (Gemini) returns correct node name ===
def router_func(state: State) -> str:
    prompt = state["input"]  # 👈 use 'input' instead of 'prompt'
    response = model.generate_content(f"""
    Classify this task: '{prompt}'
    Respond with one word only: preprocessing, ml_modeling, or reporting.
    If unknown, respond with default.
    """)
    result = response.text.strip().lower()

    if "preprocessing" in result:
        return "preprocessing"
    elif "ml_modeling" in result or "model" in result:
        return "ml_modeling"
    elif "reporting" in result:
        return "reporting"
    else:
        return "default"

# === Connect the graph ===
graph.set_conditional_entry_point(router_func)

# ✅ Compile the graph
workflow = graph.compile()

# ✅ Run test
if __name__ == "__main__":
    test_state = {"input": "Please generate a detailed report of the sales"}
    result = workflow.invoke(test_state)
    print(result)
