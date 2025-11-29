# src/graph.py (Corrected)
from langgraph.graph import StateGraph, END
from .state import AgentState # 修正相對匯入
from .agents.router import router_node # 修正相對匯入
from .agents.data_analyst import data_analyst_node # 修正相對匯入
from .agents.news_analyst import news_analyst_node # 修正相對匯入
from .agents.technical_specialist import technical_specialist_node # 導入新增的智能體
from .agents.risk_manager import risk_manager_node # 確保名稱為 risk_manager_node
from .agents.editor import editor_node # 修正相對匯入

def create_graph():
    """
    Creates the Dual-Driven Intelligent Strategy System Graph.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("data_analyst", data_analyst_node) 
    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("technical_specialist", technical_specialist_node) 
    workflow.add_node("risk_manager", risk_manager_node) # 節點名稱應匹配導入的函式
    workflow.add_node("editor", editor_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    # Router -> Data Analyst AND News Analyst (Parallel)
    workflow.add_edge("router", "data_analyst")
    workflow.add_edge("router", "news_analyst")

    # Data Analyst (獲取數據) -> Technical Specialist (分析數據)
    workflow.add_edge("data_analyst", "technical_specialist")
    
    # Risk Manager 節點等待所有分析完成
    workflow.add_edge("data_analyst", "risk_manager")
    workflow.add_edge("news_analyst", "risk_manager")
    workflow.add_edge("technical_specialist", "risk_manager") 

    # Risk Manager -> Editor
    workflow.add_edge("risk_manager", "editor")

    # Editor -> End
    workflow.add_edge("editor", END)

    return workflow.compile()