from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from ..state import AgentState
from ..utils import get_llm

class RouterOutput(BaseModel):
    tickers: List[str] = Field(description="List of stock tickers mentioned in the query")

def router_node(state: AgentState):
    """
    Router agent that extracts tickers from the user query.
    """
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(RouterOutput)
    
    system_prompt = """You are a financial assistant router. 
    Your job is to extract stock tickers from the user's query.
    If no specific ticker is mentioned, but a company name is, try to infer the ticker.
    Return a list of tickers found.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"query": state["query"]})
    
    return {"tickers": result.tickers}
