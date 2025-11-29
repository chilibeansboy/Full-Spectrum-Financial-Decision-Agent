# src/tools/search_tools.py
from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def duckduckgo_search(query: str) -> str:
    """
    Searches the web for news and general information relevant to the query 
    using the DuckDuckGo Search API. Returns a maximum of 5 recent search results.
    """
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        
    if not results:
        return f"No results found for query: {query}"
        
    # Format results
    formatted_results = []
    for i, res in enumerate(results):
        formatted_results.append(f"{i+1}. Title: {res.get('title')}\n   Snippet: {res.get('snippet')}\n   URL: {res.get('href')}")
        
    return "\n---\n".join(formatted_results)