# src/state.py
from typing import TypedDict, List, Optional, Annotated
import operator
from pandas import DataFrame # 新增導入

class AgentState(TypedDict):
    query: str
    tickers: List[str]
    
    # 原有欄位更名與調整
    fundamental_data: Optional[str] # 原 data_analysis，儲存基本面分析報告
    news_analysis: Optional[str]   # 儲存新聞分析報告
    risk_assessment: Optional[str] # 儲存風險評估報告
    final_report: Optional[str]    # 儲存最終投資備忘錄
    
    # === 專案二整合新增欄位 ===
    kline_data: DataFrame          # 儲存原始 OHLCV 數據供 Technical Specialist 使用
    technical_report: Optional[str] # Technical Specialist 產出的技術分析報告
    strategy_advice: Optional[str] # 綜合策略建議 (由 Risk Manager 輸出)
    # ===========================