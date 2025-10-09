# learning_api.py
# FastAPI integration for the learning workflow

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.ai_agents import create_learning_graph, LearningState
from langchain_core.messages import HumanMessage
import logging
from fastapi import APIRouter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent_router = APIRouter()

# Request models
class LearningRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    collection_name: Optional[str] = "learning_notes"

class TopicRequest(BaseModel):
    topics: List[str]
    difficulty: str = "beginner"

class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5

class FeedbackRequest(BaseModel):
    topic: str
    understanding: str  # good/needs_improvement
    feedback: str

# Response models
class LearningResponse(BaseModel):
    response: str
    topics: List[str]
    learning_path: List[str]
    next_step: str

# Global graph instance (in production, use proper state management)
graph = create_learning_graph()

@app.post("/api/learn", response_model=LearningResponse)
async def start_learning(request: LearningRequest):
    """
    Start a learning session with a query.
    """
    try:
        logger.info(f"Starting learning session: {request.query}")
        
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "query": request.query,
            "topics": [],
            "learning_path": [],
            "quiz_results": [],
            "summary": "",
            "current_topic": "",
            "next_step": "summary"
        }
        
        result = graph.invoke(initial_state)
        
        last_message = result['messages'][-1].content if result['messages'] else ""
        
        return LearningResponse(
            response=last_message,
            topics=result.get('topics', []),
            learning_path=result.get('learning_path', []),
            next_step=result.get('next_step', 'end')
        )
    
    except Exception as e:
        logger.error(f"Error in learning session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/teach/{topic}")
async def teach_topic(topic: str, collection_name: str = "learning_notes"):
    """
    Teach a specific topic using RAG.
    """
    try:
        logger.info(f"Teaching topic: {topic}")
        
        query = f"Teach me about {topic} using the notes from {collection_name}"
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "topics": [topic],
            "learning_path": [topic],
            "quiz_results": [],
            "summary": "",
            "current_topic": topic,
            "next_step": "teacher"
        }
        
        result = graph.invoke(initial_state)
        last_message = result['messages'][-1].content if result['messages'] else ""
        
        return {"topic": topic, "teaching": last_message}
    
    except Exception as e:
        logger.error(f"Error teaching topic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    """
    Generate a quiz for a topic.
    """
    try:
        logger.info(f"Generating quiz for: {request.topic}")
        
        query = f"Generate a {request.num_questions} question quiz on {request.topic}"
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "topics": [request.topic],
            "learning_path": [],
            "quiz_results": [],
            "summary": "",
            "current_topic": request.topic,
            "next_step": "quiz"
        }
        
        result = graph.invoke(initial_state)
        last_message = result['messages'][-1].content if result['messages'] else ""
        
        return {"topic": request.topic, "quiz": last_message}
    
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mentor/feedback")
async def get_mentor_feedback(request: FeedbackRequest):
    """
    Get mentor feedback on understanding.
    """
    try:
        logger.info(f"Getting mentor feedback for: {request.topic}")
        
        query = f"Review my understanding of {request.topic}. My current level: {request.understanding}. {request.feedback}"
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "topics": [request.topic],
            "learning_path": [],
            "quiz_results": [],
            "summary": "",
            "current_topic": request.topic,
            "next_step": "mentor"
        }
        
        result = graph.invoke(initial_state)
        last_message = result['messages'][-1].content if result['messages'] else ""
        
        return {"topic": request.topic, "feedback": last_message}
    
    except Exception as e:
        logger.error(f"Error getting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plan")
async def create_learning_plan(request: TopicRequest):
    """
    Create a learning plan from topics.
    """
    try:
        logger.info(f"Creating learning plan for: {request.topics}")
        
        topics_str = ", ".join(request.topics)
        query = f"Create a {request.difficulty} learning plan for: {topics_str}"
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "topics": request.topics,
            "learning_path": [],
            "quiz_results": [],
            "summary": "",
            "current_topic": "",
            "next_step": "planner"
        }
        
        result = graph.invoke(initial_state)
        
        return {
            "topics": result.get('topics', request.topics),
            "learning_path": result.get('learning_path', []),
            "plan": result['messages'][-1].content if result['messages'] else ""
        }
    
    except Exception as e:
        logger.error(f"Error creating plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "learning-workflow"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Learning Workflow API",
        "version": "1.0",
        "endpoints": {
            "learn": "/api/learn",
            "teach": "/api/teach/{topic}",
            "quiz": "/api/quiz",
            "mentor": "/api/mentor/feedback",
            "plan": "/api/plan"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)