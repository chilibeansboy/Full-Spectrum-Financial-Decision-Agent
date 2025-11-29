# src/agents/risk_manager.py (Corrected)
from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState # 修正相對匯入
from ..utils import get_llm # 修正相對匯入

def risk_manager_node(state: AgentState):
    """
    Strategy & Risk Manager that synthesizes all reports and provides actionable advice.
    """
    llm = get_llm(temperature=0)
    
    system_prompt = """You are the Chief Investment Strategist and Risk Manager at a major investment fund.
    您的工作是綜合來自基本面分析師、新聞分析師和技術分析專家的報告，制定出一個最終的、可執行的投資策略，**特別針對用戶的提問**。
    
    您的主要輸出必須是一個統一的策略，包含：
    1. **Strategy Advice (策略建議)**: 給出明確的建議 (看多/買入, 看空/賣出, 或 持有/觀望)，並解釋您的理由，同時解決基本面和技術面之間可能存在的衝突信號。
    2. **Actionable Parameters (具體交易參數)**: 根據技術分析報告，建議具體的**進場價位 (Entry)、止損價位 (SL) 和止盈價位 (TP) 價格水平**，用於短期交易。
    3. **Overall Risks (總體風險)**: 最終評估下行風險 (監管、競爭) 和技術波動性。
    
    請保持審慎。如果股票「定價完美」，請將此作為主要風險點強調出來。
    請以 **Traditional Chinese (繁體中文)** 輸出您的完整綜合策略和風險評估。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """User Query:
{user_query}

--- ALL RESEARCH REPORTS ---

1. Fundamental Analysis:
{fundamental_data}

2. News Analysis:
{news_analysis}

3. Technical Analysis:
{technical_report}

請提供您的綜合投資策略和詳細的風險評估。""")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "user_query": state.get("query", "未提供具體查詢。"),
        "fundamental_data": state.get("fundamental_data", "未提供基本面分析報告。"),
        "news_analysis": state.get("news_analysis", "未提供新聞分析報告。"),
        "technical_report": state.get("technical_report", "未提供技術分析報告。"), 
    })
    
    # 確保返回 dictionary 的鍵名與 src/state.py 中的定義匹配
    return {
        "risk_assessment": response.content,
        "strategy_advice": response.content 
    }