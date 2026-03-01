import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json

from agent.state import AgentState
from tools.script_builder import save_script

def generate_script_node(state: AgentState) -> AgentState:
    print("[generate_script] Generating Python script from paper info...")

    prompt_template = Path()