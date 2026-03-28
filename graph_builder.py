from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from tools import get_weather, multiply, get_fun_fact

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [get_weather, multiply, get_fun_fact]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)
