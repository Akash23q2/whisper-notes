from langgraph.graph import StateGraph
from app.services.agent_service import planner_agent, quiz_agent, teacher_agent, mentor_agent, basic_agent, AgentState
from langgraph.graph import START, END, StateGraph
workflow = StateGraph(AgentState)

workflow.add_node("planner_agent", planner_agent)
workflow.add_node("quiz_agent", quiz_agent)
workflow.add_node("teacher_agent", teacher_agent)
workflow.add_node("mentor_agent", mentor_agent)
workflow.add_node("basic_agent", basic_agent)

workflow.add_edge(START, "planner_agent")
workflow.add_conditional_edges(
    "planner_agent",
    lambda state: state.get("next_agent"),
    {
        "quiz_agent": "quiz_agent",
        "teacher_agent": "teacher_agent",
        "mentor_agent": "mentor_agent",
        "basic_agent": "basic_agent"
    }
)
workflow.add_edge("quiz_agent", END)
workflow.add_edge("teacher_agent", END)
workflow.add_edge("mentor_agent", END)
workflow.add_edge("basic_agent", END)

graph = workflow.compile()
