
from flask import Flask, render_template, request
import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

app = Flask(__name__)

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "Enter_Your_Gemini_Key"

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)

def classify_problem(state : dict ) -> dict:
    prompt = (
        "You are a helpful environmental assistant. Classify the user's concern below into one of the categories:\n"
        "- Awareness\n- Action\n- Emergency\n\n"
        f"problem: {state['problem']}\n"
        "Respond with only one word: Awareness, Action, or Emergency.\n"
        "#Example: input: There is smoke in the sky from nearby fires. → output: Emergency"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    state["category"] = response.content.strip()
    return state

def problem_router(state:dict) -> str:
    cat = state["category"].lower()
    if "awareness" in cat:
        return "awareness"
    elif "emergency" in cat:
        return "emergency"
    elif "action" in cat:
        return "action"
    return "awareness"

def awareness_node(state: dict) -> dict:
    prompt = f"You are a knowledgeable climate educator.\nUser's question: {state['problem']}\nGive an informative, clear explanation to raise the user's awareness about this issue."
    response = llm.invoke([HumanMessage(content=prompt)])
    state["answer"] = response.content.strip()
    return state

def emergency_node(state: dict) -> dict:
    prompt = f"You are an environmental emergency assistant.\nSituation: {state['problem']}\nGive urgent advice or immediate steps the user should take for their safety."
    response = llm.invoke([HumanMessage(content=prompt)])
    state["answer"] = response.content.strip()
    return state

def action_node(state: dict) -> dict:
    prompt = f"You are a helpful environmental assistant.\nUser's request: {state['problem']}\nPlease provide a specific, actionable response to help the user take positive climate action based on their concern."
    response = llm.invoke([HumanMessage(content=prompt)])
    state["answer"] = response.content.strip()
    return state

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        problem = request.form["problem"]
        state = {"problem": problem}
        classify_problem(state)
        category = problem_router(state)
        if category == "awareness":
            state = awareness_node(state)
        elif category == "emergency":
            state = emergency_node(state)
        elif category == "action":
            state = action_node(state)
        response = state.get("answer")
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
