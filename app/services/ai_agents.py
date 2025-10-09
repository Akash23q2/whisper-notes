# learning_workflow.py
from typing import TypedDict, Annotated, Sequence, Literal, List
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from app.services.agent_tools import tools

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# State definition
class LearningState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    summary: str
    topics: List[str]
    current_topic: str
    learning_path: List[str]
    quiz_results: List[dict]
    next_step: str

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Agent Node Functions
def get_summary_node(state: LearningState):
    """Get summary of data using embeddings and batch processing"""
    msgs = state['messages']
    query = state['query']
    
    system_prompt = """You are a data summarizer. Use the getConversationSummary and 
    retrieveFromEmbeddings tools to gather context and create a comprehensive summary.
    Extract all key topics mentioned in the conversation."""
    
    msgs_with_system = [SystemMessage(content=system_prompt)] + list(msgs)
    response = llm_with_tools.invoke(msgs_with_system)
    
    return {
        "messages": [response],
        "next_step": "planner"
    }

def planner_agent(state: LearningState):
    """Plan learning path and extract topics"""
    msgs = state['messages']
    
    system_prompt = """You are a learning path planner. Based on the summary and user query:
    1. List ALL topics that need to be covered
    2. Create a personalized learning path (order of topics)
    3. Store topics in a structured format
    
    Output format:
    TOPICS: [topic1, topic2, topic3]
    PATH: [step1, step2, step3]
    """
    
    msgs_with_system = [SystemMessage(content=system_prompt)] + list(msgs)
    response = llm_with_tools.invoke(msgs_with_system)
    
    # Extract topics from response
    content = response.content
    topics = []
    learning_path = []
    
    if "TOPICS:" in content:
        topics_line = content.split("TOPICS:")[1].split("PATH:")[0].strip()
        topics = [t.strip() for t in topics_line.strip("[]").split(",")]
    
    if "PATH:" in content:
        path_line = content.split("PATH:")[1].strip()
        learning_path = [p.strip() for p in path_line.strip("[]").split(",")]
    
    return {
        "messages": [response],
        "topics": topics,
        "learning_path": learning_path,
        "next_step": "router"
    }

def basic_agent(state: LearningState):
    """Answer general queries with context"""
    msgs = state['messages']
    
    system_prompt = """You are a helpful assistant. Answer the user's query based on:
    - General knowledge
    - Conversation context (use getConversationSummary if needed)
    - Available tools for current information"""
    
    msgs_with_system = [SystemMessage(content=system_prompt)] + list(msgs)
    response = llm_with_tools.invoke(msgs_with_system)
    
    return {
        "messages": [response],
        "next_step": "end"
    }

def teacher_agent(state: LearningState):
    """Teach a topic using RAG and conversational manner"""
    msgs = state['messages']
    current_topic = state.get('current_topic', '')
    
    system_prompt = f"""You are an expert teacher. Teach the topic: {current_topic}
    
    Instructions:
    1. Use retrieveFromEmbeddings to get relevant information from notes
    2. Use searchWebTool for current information if needed
    3. Teach in a conversational, engaging manner
    4. Provide examples and explanations
    5. Use augmented generation (RAG + your knowledge)
    
    Make it interactive and easy to understand."""
    
    msgs_with_system = [SystemMessage(content=system_prompt)] + list(msgs)
    response = llm_with_tools.invoke(msgs_with_system)
    
    return {
        "messages": [response],
        "next_step": "mentor"
    }

def mentor_agent(state: LearningState):
    """Verify understanding and find strong/weak points"""
    msgs = state['messages']
    current_topic = state.get('current_topic', '')
    
    system_prompt = f"""You are a mentor reviewing understanding of: {current_topic}
    
    Tasks:
    1. Verify if the student's responses are correct
    2. Use retrieveFromEmbeddings to check against source material
    3. Review conversation history (getConversationSummary)
    4. Identify strong points and weak points
    5. Provide constructive feedback
    
    Output format:
    UNDERSTANDING: [good/needs_improvement]
    STRONG_POINTS: [list]
    WEAK_POINTS: [list]
    RECOMMENDATION: [next steps]"""
    
    msgs_with_system = [SystemMessage(content=system_prompt)] + list(msgs)
    response = llm_with_tools.invoke(msgs_with_system)
    
    return {
        "messages": [response],
        "next_step": "planner"
    }

def quiz_agent(state: LearningState):
    """Generate quiz based on topics"""
    msgs = state['messages']
    topics = state.get('topics', [])
    current_topic = state.get('current_topic', '')
    
    topic_str = current_topic if current_topic else ", ".join(topics[:3])
    
    system_prompt = f"""You are a quiz generator. Create a quiz on: {topic_str}
    
    Requirements:
    1. Generate 5 questions (multiple choice and short answer)
    2. Questions should test understanding, not memorization
    3. Include varying difficulty levels
    4. Provide correct answers at the end
    
    Use retrieveFromEmbeddings to ensure questions align with taught content."""
    
    msgs_with_system = [SystemMessage(content=system_prompt)] + list(msgs)
    response = llm_with_tools.invoke(msgs_with_system)
    
    return {
        "messages": [response],
        "next_step": "end"
    }

# Tool execution node
tool_node = ToolNode(tools)

def should_continue(state: LearningState) -> str:
    """Route based on tool calls"""
    last_msg = state['messages'][-1]
    
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    
    return state.get('next_step', 'end')

def router(state: LearningState) -> str:
    """Route to appropriate agent based on context"""
    msgs = state['messages']
    last_msg = msgs[-1].content if msgs else ""
    topics = state.get('topics', [])
    
    # Check if user wants quiz
    if any(word in last_msg.lower() for word in ['quiz', 'test', 'assess']):
        return "quiz"
    
    # Check if teaching is needed
    if topics and any(word in last_msg.lower() for word in ['teach', 'learn', 'explain']):
        # Set first topic as current
        if not state.get('current_topic'):
            state['current_topic'] = topics[0]
        return "teacher"
    
    # Default to basic agent
    return "basic"

# Build the graph
def create_learning_graph():
    workflow = StateGraph(LearningState)
    
    # Add nodes
    workflow.add_node("summary", get_summary_node)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("basic", basic_agent)
    workflow.add_node("teacher", teacher_agent)
    workflow.add_node("mentor", mentor_agent)
    workflow.add_node("quiz", quiz_agent)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "summary")
    
    # Summary can call tools or go to planner
    workflow.add_conditional_edges(
        "summary",
        should_continue,
        {
            "tools": "tools",
            "planner": "planner",
            "end": END
        }
    )
    
    # Planner routes to different agents
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "tools": "tools",
            "router": "router",
            "end": END
        }
    )
    
    # Router decides which agent to use
    workflow.add_node("router", lambda s: s)  # Pass-through node
    workflow.add_conditional_edges(
        "router",
        router,
        {
            "basic": "basic",
            "teacher": "teacher",
            "quiz": "quiz"
        }
    )
    
    # Teacher can call tools or go to mentor
    workflow.add_conditional_edges(
        "teacher",
        should_continue,
        {
            "tools": "tools",
            "mentor": "mentor",
            "end": END
        }
    )
    
    # Mentor can call tools or go back to planner
    workflow.add_conditional_edges(
        "mentor",
        should_continue,
        {
            "tools": "tools",
            "planner": "planner",
            "end": END
        }
    )
    
    # Quiz and basic agent edges
    workflow.add_conditional_edges(
        "quiz",
        should_continue,
        {"tools": "tools", "end": END}
    )
    
    workflow.add_conditional_edges(
        "basic",
        should_continue,
        {"tools": "tools", "end": END}
    )
    
    # Tools always return to the agent that called them
    workflow.add_conditional_edges(
        "tools",
        lambda s: s.get('next_step', 'planner')
    )
    
    return workflow.compile()

# Usage example
def run_learning_workflow(user_query: str):
    """Run the learning workflow"""
    graph = create_learning_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "query": user_query,
        "topics": [],
        "learning_path": [],
        "quiz_results": [],
        "summary": "",
        "current_topic": "",
        "next_step": "summary"
    }
    
    result = graph.invoke(initial_state)
    return result

# API endpoint integration
if __name__ == "__main__":
    # Test the workflow
    result = run_learning_workflow("I want to learn about machine learning basics")
    print("Final Messages:", result['messages'][-1].content)