# src/agents/technical_specialist.py (New File)
from langchain_core.prompts import ChatPromptTemplate
from ..utils import get_llm
from ..tools.finance_tools import calculate_technical_indicators # 修正相對匯入
from ..state import AgentState # 修正相對匯入

TECH_ANALYSIS_PROMPT = """
You are the Technical Analysis Specialist. Your task is to analyze the provided raw technical indicator calculations and the recent K-line data to identify chart patterns and market trends.

您必須在報告中提供兩個關鍵部分，並以**Traditional Chinese (繁體中文)**呈現：

1. **Indicator Interpretation (指標解讀)**: 分析 RSI, MACD Histogram 和 Stochastic Oscillator 的數值，評估動量和超買/超賣情況。
2. **Trend & Pattern (趨勢與形態)**: 根據價格數據（K 線快照）固有的趨勢和形態，明確指出一個清晰的形態（例如：雙底、盤整區間）和一個主要的價格趨勢（上升、下降或橫盤）。

請生成結構化的技術分析報告 (Markdown 格式)。

---
**Raw Technical Indicator Calculation Results (原始技術指標計算結果):**
{indicator_calculation_result}

**Recent K-line Data Snapshot (最近 K 線數據快照 - Last 5 Days):**
{kline_snapshot}
"""

def technical_specialist_node(state: AgentState) -> dict:
    """
    Analyzes raw K-line data, calculates indicators, and identifies technical trends/patterns.
    """
    # 檢查 kline_data 是否存在
    if 'kline_data' not in state or state['kline_data'].empty:
         return {"technical_report": "技術分析已跳過：未找到足夠的 K 線數據。"}

    llm = get_llm(temperature=0) 

    # 1. 計算原始指標
    kline_df = state['kline_data']
    indicator_result = calculate_technical_indicators(kline_df)

    # 2. 準備 K 線數據快照 (最後 5 天) 給 LLM 參考
    kline_snapshot = kline_df.tail(5).to_markdown()

    # 3. 執行 LLM 鏈進行解讀和趨勢/形態分析
    prompt = ChatPromptTemplate.from_template(TECH_ANALYSIS_PROMPT)
    
    chain = prompt | llm 

    analysis_result = chain.invoke({
        "indicator_calculation_result": indicator_result,
        "kline_snapshot": kline_snapshot
    })

    # 4. 更新狀態
    return {"technical_report": analysis_result.content}