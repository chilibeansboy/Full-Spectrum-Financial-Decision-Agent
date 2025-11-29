# src/agents/editor.py
from langchain_core.prompts import ChatPromptTemplate
from ..state import AgentState # 修正相對匯入
from ..utils import get_llm # 修正相對匯入

def editor_node(state: AgentState):
    """
    Chief Editor that compiles the final investment memo.
    """
    llm = get_llm(temperature=0)
    
    system_prompt = """You are the Chief Editor of a prestigious investment research firm (like Goldman Sachs or Morgan Stanley).
    您的目標是將所有生成的研究報告編譯成一份連貫、專業的「賣方」投資報告，**特別針對用戶的提問**。
    
    輸入包含：
    - User Query
    - Fundamental Analysis
    - News Analysis
    - Technical Analysis
    - Integrated Strategy & Risk Assessment
    
    輸出：
    - 一份專業的 Markdown 報告，以 **Traditional Chinese (繁體中文)** 呈現。
    - **風格規則**: 權威、專業、果斷。論點必須以數據支持。
    
    報告結構:
    
    1. **Executive Summary & Strategy (執行摘要與策略)**:
        - **Direct Answer**: 一個敘述段落，明確回應用戶的提問。
        - **Rating & Target**: 評級 (買入/持有/賣出) 和目標價。
        - **Integrated Strategy**: 核心投資論點，結合基本面和技術面。
    
    2. **Fundamental Analysis (基本面分析)**:
        - 敘述性分析估值、質量和成長性。
    
    3. **Technical Analysis & Trading Parameters (技術分析與交易參數)**:
        - 總結短期價格動能、圖表形態和技術指標。
        - **具體交易參數**：列出建議的進場價位、止損價位和止盈價位。
    
    4. **News & Market Sentiment (新聞與市場情緒)**:
        - 分析關鍵催化劑和市場基調。
    
    5. **Risk Factors & Conclusion (風險因素與結論)**: 最終風險評估和總結。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """User Query:
{user_query}

Fundamental Analysis:
{fundamental_data}

News Analysis:
{news_analysis}

Technical Analysis:
{technical_report}

Strategy Advice & Risks:
{strategy_advice}

請生成最終的投資備忘錄。""")
    ])
    
    chain = prompt | llm
    
    # 傳入所有報告
    response = chain.invoke({
        "user_query": state.get("query", "未提供具體查詢。"),
        "fundamental_data": state.get("fundamental_data"), # 使用新名稱
        "news_analysis": state.get("news_analysis"),
        "technical_report": state.get("technical_report"), # 新增欄位
        "strategy_advice": state.get("strategy_advice") # 新增欄位
    })
    
    return {"final_report": response.content}