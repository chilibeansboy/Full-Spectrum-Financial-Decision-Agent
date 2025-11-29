# src/agents/data_analyst.py
from langchain.agents import create_agent
from ..state import AgentState # 修正相對匯入
from ..tools.finance_tools import fetch_fundamental_and_kline_data # 修正匯入
from ..utils import get_llm # 修正相對匯入
from langchain_core.prompts import ChatPromptTemplate # 新增導入
# 移除 create_agent 的導入，因為此節點現在是純 LLM chain

def data_analyst_node(state: AgentState):
    """
    Finance Data Analyst that gathers fundamental data and prepares K-line data
    for the Technical Specialist.
    """
    llm = get_llm(temperature=0)
    
    # 1. 執行數據獲取和 K-line 數據準備
    ticker = state["tickers"][0]
    fundamental_text, kline_df = fetch_fundamental_and_kline_data(ticker)

    # 2. 執行基本面分析 (LLM 部分)
    system_prompt = """You are a Senior Financial Data Analyst at a top-tier investment bank.
    您的目標是提供對所提供股票代碼的嚴謹量化分析，**特別針對用戶的提問**。
    
    以下數據已編譯供您分析：
    {data_input}
    
    您的報告必須著重於：
    1. **Valuation Analysis (估值分析)**: 比較 P/E, PEG 和 EV/EBITDA 等與歷史或市場基準。股票是便宜還是昂貴？
    2. **Financial Health (財務狀況)**: 分析毛利率/營運利潤率、營收/盈利增長率，以及資產負債表強度。
    3. **Analyst Consensus (分析師共識)**: 總結市場觀點（目標價、評級）。
    
    請輸出結構化的分析報告，以 **Traditional Chinese (繁體中文)** 呈現。請解釋數字而非僅僅列出數字。
    - **Direct Answer to User (針對用戶問題的數據回應)**
    - **Valuation Verdict (估值判斷)**: 低估 / 合理 / 高估 (需有依據)。
    - **Quality Score (品質評分)**: 高 / 中 / 低 (基於利潤率和 ROE)。
    - **Growth Outlook (成長展望)**: 強勁 / 適中 / 疲弱。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(data_input=fundamental_text)),
        ("human", f"基於提供的數據，分析 {ticker} 的核心財務健康和估值，以回應用戶的問題： {state['query']}"),
    ])
    
    chain = prompt | llm
    analysis_result = chain.invoke({})
    
    # 3. 更新狀態
    return {
        "fundamental_data": analysis_result.content,
        "kline_data": kline_df  
    }