# src/agents/news_analyst.py (Corrected)
# 移除 from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState # 修正相對匯入
from ..tools.search_tools import duckduckgo_search # 修正匯入的函式名稱
from ..utils import get_llm # 修正相對匯入

def news_analyst_node(state: AgentState):
    """
    Finance News Analyst that gathers and analyzes recent news and market sentiment.
    """
    llm = get_llm(temperature=0)
    
    ticker = state['tickers'][0]
    query = state['query']
    
    # 1. 執行 Tool 呼叫獲取原始搜索結果
    # 由於原專案的 design pattern，我們在這裡直接呼叫 Tool 函式
    search_results = duckduckgo_search(f"Recent news and market sentiment for {ticker} relating to {query}") 
    
    # 2. LLM Synthesis Prompt
    system_prompt = """您是金融新聞分析師。您的任務是根據以下原始搜索結果，綜合整理最相關的新聞、盈利報告和市場情緒。
    
    您的報告必須著重於：
    1. 總結關鍵的催化劑和市場反應。
    2. 指出任何重大的非財務風險（如監管、法律問題）。
    
    請輸出結構化的新聞分析報告，以 **Traditional Chinese (繁體中文)** 呈現。
    - **Key Catalysts (關鍵催化劑)**: 列出 1-3 個最近最重要的事件。
    - **Sentiment Summary (情緒總結)**: 簡短段落說明市場基調 (正向/負向/中性)。
    - **Non-Financial Risks (非財務風險)**: 指出任何法規或政治風險。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Search Results:\n---\n{search_results}\n---\n請綜合這些資訊並生成報告。"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({})

    # 確保輸出到狀態欄位 news_analysis
    return {"news_analysis": response.content}