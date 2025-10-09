"""
Agent Service - Gamified Learning Platform
Implements agentic workflow using LangGraph for interactive learning
"""

from typing import TypedDict, Annotated, Sequence, Literal, List, Optional
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from app.utils.tools import (
    searchWebTool,
    searchNewsTool,
    getPresentEmbeddingsInfo,
    createEmbeddingsFromText,
    createEmbeddingsFromPDF,
    createEmbeddingsFromURL,
    retrieveFromEmbeddings,
    getCurrentDateTime,
    logConversation,
    getConversationSummary
)
from app.schemas.agent_schema import QuizResults, GameCharacters, Topics
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

## State definition
class LearningState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    session_summary: str
    topics: List[Topics]
    characters: List[GameCharacters]
    current_topic: str
    current_character: Optional[GameCharacters]
    quiz_results: List[QuizResults]
    next_step: Literal["planner", "teacher", "quiz_generator", "evaluator", "end"]
    learning_context: str  # Retrieved RAG context
    user_progress: dict  # Track completed topics and scores

## Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", #use flash 2.0 as 2.5 doesnt support code execution yet
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

## Tools available to the agent
tools = [
    searchWebTool,
    searchNewsTool,
    getPresentEmbeddingsInfo,
    retrieveFromEmbeddings,
    getCurrentDateTime,
    logConversation,
    getConversationSummary,
    createEmbeddingsFromText,
    createEmbeddingsFromPDF,
    createEmbeddingsFromURL,
]

llm_with_tools = llm.bind_tools(tools)

## Agent Implementation
def planner_agent(state: LearningState) -> LearningState:
    """
    Planner agent: Analyzes available embeddings, extracts topics,
    generates game characters, and plans the learning journey
    """
    messages = state["messages"]
    
    # Get available collections info
    collections_info = getPresentEmbeddingsInfo.invoke({})
    
    system_prompt = f"""You are a Learning Journey Planner for a gamified education platform.
    
Available learning materials in database:
{collections_info}

Your tasks:
1. Analyze the available content and extract key topics
2. Create engaging game characters (NPCs) for each topic
3. Generate a learning sequence that progressively builds knowledge
4. Each character should have a unique personality and teaching style

Return a structured response with:
- List of topics with descriptions
- List of game characters with their associated topics
- Recommended learning sequence

Be creative and make the learning journey exciting!"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])
    
    # Parse the response to extract topics and characters
    # In production, you'd use structured output or JSON parsing
    state["messages"] = state["messages"] + [response]
    state["session_summary"] = f"Learning journey planned with topics from: {collections_info}"
    
    return state


def teacher_agent(state: LearningState) -> LearningState:
    """
    Teacher agent: Retrieves relevant content and teaches the current topic
    Uses RAG to provide contextual, accurate information
    """
    current_topic = state.get("current_topic", "")
    current_character = state.get("current_character")
    messages = state["messages"]
    
    # Get available collections
    collections_info = getPresentEmbeddingsInfo.invoke({})
    
    # Retrieve relevant context from embeddings
    # Try to find the most relevant collection
    collection_name = state.get("current_collection", "default")
    
    try:
        rag_context = retrieveFromEmbeddings.invoke({
            "collection_name": collection_name,
            "query": current_topic,
            "n_results": 5
        })
        state["learning_context"] = rag_context
    except Exception as e:
        rag_context = f"Could not retrieve specific context: {e}"
        state["learning_context"] = ""
    
    # Get conversation history for context
    conversation_summary = ""
    try:
        conversation_summary = getConversationSummary.invoke({
            "query": current_topic,
            "n_results": 5
        })
    except:
        pass
    
    character_intro = ""
    if current_character:
        character_intro = f"""
You are {current_character.name}, a character in a learning game.
Your role: {current_character.description}
Your expertise: {current_character.topic}
"""
    
    system_prompt = f"""{character_intro}

You are teaching about: {current_topic}

Relevant learning materials:
{rag_context}

Previous conversation context:
{conversation_summary}

Your teaching approach:
1. Start with a warm, engaging introduction
2. Break down complex concepts into digestible pieces
3. Use examples, analogies, and real-world applications
4. Check for understanding with questions
5. Be encouraging and supportive
6. Stay in character and make learning fun!

Teach the topic thoroughly but conversationally. Encourage questions."""

    response = llm_with_tools.invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])
    
    # Log the conversation
    if messages:
        last_user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        logConversation.invoke({
            "user_message": last_user_msg,
            "ai_response": response.content
        })
    
    state["messages"] = state["messages"] + [response]
    
    return state


def quiz_generator_agent(state: LearningState) -> LearningState:
    """
    Quiz generator: Creates questions based on what was taught
    Generates MCQs with varying difficulty levels
    """
    current_topic = state.get("current_topic", "")
    learning_context = state.get("learning_context", "")
    messages = state["messages"]
    
    system_prompt = f"""You are a Quiz Master creating an assessment for: {current_topic}

Learning materials covered:
{learning_context}

Create a quiz with 5 multiple-choice questions that:
1. Test understanding of key concepts
2. Range from basic recall to application
3. Include 4 options each (A, B, C, D)
4. Have clear, unambiguous correct answers
5. Provide explanations for learning

Format each question as JSON:
{{
    "question": "question text",
    "mcq_options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "correct_ans": "B",
    "explanation": "why this is correct"
}}

Return an array of 5 questions."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *messages[-3:]  # Include recent context
    ])
    
    # Parse quiz questions
    # In production, use structured output
    state["messages"] = state["messages"] + [
        AIMessage(content=f"Quiz generated for {current_topic}. Ready to test your knowledge!"),
        response
    ]
    
    return state


def evaluator_agent(state: LearningState) -> LearningState:
    """
    Evaluator: Assesses quiz answers, provides feedback,
    and determines if student can progress
    """
    quiz_results = state.get("quiz_results", [])
    current_topic = state.get("current_topic", "")
    messages = state["messages"]
    
    # Calculate score
    total_questions = len(quiz_results)
    correct_answers = sum(1 for q in quiz_results if q.user_ans.upper() == q.correct_ans.upper())
    score_percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    
    # Determine if student passes (70% threshold)
    passed = score_percentage >= 70
    
    system_prompt = f"""You are an Encouraging Evaluator reviewing quiz performance.

Topic: {current_topic}
Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)
Status: {"PASSED ✓" if passed else "NEEDS REVIEW"}

Quiz Results:
{json.dumps([{
    "question": q.question,
    "user_answer": q.user_ans,
    "correct_answer": q.correct_ans,
    "correct": q.user_ans.upper() == q.correct_ans.upper()
} for q in quiz_results], indent=2)}

Provide:
1. Encouraging feedback on performance
2. Detailed explanation of incorrect answers
3. Suggestions for improvement on weak areas
4. {"Congratulations and next steps" if passed else "Encouragement to review and retry"}

Be supportive, specific, and constructive!"""

    response = llm.invoke([
        SystemMessage(content=system_prompt)
    ])
    
    # Update user progress
    if "user_progress" not in state:
        state["user_progress"] = {}
    
    state["user_progress"][current_topic] = {
        "score": score_percentage,
        "passed": passed,
        "attempts": state["user_progress"].get(current_topic, {}).get("attempts", 0) + 1
    }
    
    state["messages"] = state["messages"] + [response]
    
    # Determine next step
    if passed:
        state["next_step"] = "planner"  # Move to next topic
    else:
        state["next_step"] = "teacher"  # Review the topic
    
    return state


def router(state: LearningState) -> Literal["planner", "teacher", "quiz_generator", "evaluator", "tools", "end"]:
    """
    Router: Determines the next node based on current state
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if there are tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Check explicit next_step
    next_step = state.get("next_step", "")
    if next_step == "end":
        return "end"
    elif next_step == "teacher":
        return "teacher"
    elif next_step == "quiz_generator":
        return "quiz_generator"
    elif next_step == "evaluator":
        return "evaluator"
    elif next_step == "planner":
        return "planner"
    
    # Default logic based on conversation state
    if not state.get("topics"):
        return "planner"
    elif not state.get("quiz_results"):
        # In teaching phase
        user_wants_quiz = any("quiz" in msg.content.lower() for msg in messages[-3:] if isinstance(msg, HumanMessage))
        if user_wants_quiz:
            return "quiz_generator"
        return "teacher"
    else:
        # Has quiz results, need evaluation
        return "evaluator"


def should_continue(state: LearningState) -> Literal["continue", "end"]:
    """
    Determines if the workflow should continue or end
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for explicit end signals
    if state.get("next_step") == "end":
        return "end"
    
    # Check if user wants to exit
    if messages and isinstance(messages[-1], HumanMessage):
        exit_keywords = ["exit", "quit", "bye", "end session"]
        if any(keyword in messages[-1].content.lower() for keyword in exit_keywords):
            return "end"
    
    return "continue"


## BUILD WORKFLOW GRAPH

def create_learning_workflow():
    """
    Creates and compiles the LangGraph workflow
    """
    workflow = StateGraph(LearningState)
    
    # Add nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("teacher", teacher_agent)
    workflow.add_node("quiz_generator", quiz_generator_agent)
    workflow.add_node("evaluator", evaluator_agent)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.add_edge(START, "planner")
    
    # Conditional routing from each agent
    workflow.add_conditional_edges(
        "planner",
        router,
        {
            "planner": "planner",
            "teacher": "teacher",
            "quiz_generator": "quiz_generator",
            "evaluator": "evaluator",
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "teacher",
        router,
        {
            "planner": "planner",
            "teacher": "teacher",
            "quiz_generator": "quiz_generator",
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "quiz_generator",
        router,
        {
            "quiz_generator": "quiz_generator",
            "evaluator": "evaluator",
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "evaluator",
        router,
        {
            "planner": "planner",
            "teacher": "teacher",
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always return to the agent that called them
    workflow.add_edge("tools", "teacher")
    
    return workflow.compile()


## SERVICE CLASS

class LearningAgentService:
    """
    Main service class for the learning agent
    """
    
    def __init__(self):
        self.workflow = create_learning_workflow()
        self.current_state = None
    
    def initialize_session(self, user_query: str) -> dict:
        """
        Initialize a new learning session
        """
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "query": user_query,
            "session_summary": "",
            "topics": [],
            "characters": [],
            "current_topic": "",
            "current_character": None,
            "quiz_results": [],
            "next_step": "planner",
            "learning_context": "",
            "user_progress": {}
        }
        
        self.current_state = initial_state
        return initial_state
    
    def process_message(self, user_message: str) -> dict:
        """
        Process a user message and update state
        """
        if not self.current_state:
            return self.initialize_session(user_message)
        
        # Add user message to state
        self.current_state["messages"].append(HumanMessage(content=user_message))
        
        # Run workflow
        result = self.workflow.invoke(self.current_state)
        self.current_state = result
        
        return result
    
    def set_current_topic(self, topic: str, character: GameCharacters = None):
        """
        Set the current learning topic
        """
        if self.current_state:
            self.current_state["current_topic"] = topic
            self.current_state["current_character"] = character
            self.current_state["next_step"] = "teacher"
    
    def submit_quiz_answers(self, quiz_results: List[QuizResults]):
        """
        Submit quiz answers for evaluation
        """
        if self.current_state:
            self.current_state["quiz_results"] = quiz_results
            self.current_state["next_step"] = "evaluator"
            
            # Run evaluator
            result = self.workflow.invoke(self.current_state)
            self.current_state = result
            return result
    
    def get_progress(self) -> dict:
        """
        Get user's learning progress
        """
        if self.current_state:
            return self.current_state.get("user_progress", {})
        return {}
    
    def end_session(self) -> str:
        """
        End the learning session
        """
        if self.current_state:
            progress = self.get_progress()
            total_topics = len(progress)
            passed_topics = sum(1 for p in progress.values() if p.get("passed", False))
            
            summary = f"""
Learning Session Summary:
- Topics Covered: {total_topics}
- Topics Mastered: {passed_topics}
- Overall Progress: {(passed_topics/total_topics*100) if total_topics > 0 else 0:.1f}%

Keep up the great work!
            """
            return summary
        return "No active session"


## USAGE EXAMPLE

# if __name__ == "__main__":
#     # Example usage
#     service = LearningAgentService()
    
#     # Initialize session
#     service.initialize_session("I want to learn about thermodynamics from my uploaded notes")
    
#     # Set topic
#     character = GameCharacters(
#         name="Professor Thermo",
#         topic="Thermodynamics",
#         description="A friendly professor who teaches thermodynamics with real-world examples"
#     )
#     service.set_current_topic("Thermodynamics - First Law", character)
    
#     # Process messages
#     result = service.process_message("Can you explain the first law of thermodynamics?")
#     print(result["messages"][-1].content)