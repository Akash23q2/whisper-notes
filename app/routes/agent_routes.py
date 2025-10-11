from fastapi import APIRouter, HTTPException
from app.schemas.response_schema import AgentRequest, AgentResponse
from app.services.agent_service import run_agent_task, SupportDependencies, avilable_collections
from app.schemas.agent_schema import AgentMode

# Create router
agent_router = APIRouter(prefix="/agent", tags=["AI Agent"])


@agent_router.get("/ping")
async def ping_agent():
    """Health check for agent system."""
    return {
        "status": "active",
        "collections_count": len(avilable_collections),
        "timestamp": SupportDependencies.getCurrentDateTime()
    }


@agent_router.get("/modes")
async def get_agent_modes():
    """Get all supported agent modes with descriptions."""
    return {
        "modes": [
            {
                "name": AgentMode.PLAN,
                "description": "Plan a gamified learning journey with topics, NPCs, and quizzes"
            },
            {
                "name": AgentMode.TEACH,
                "description": "Interactive teaching with personality and examples"
            },
            {
                "name": AgentMode.NOTES,
                "description": "Generate structured notes using RAG context"
            },
            {
                "name": AgentMode.RAG,
                "description": "Answer questions using retrieved embeddings"
            }
        ]
    }


@agent_router.get("/collections")
async def get_collections():
    """Get all available RAG collections."""
    return {
        "collections": avilable_collections,
        "info": SupportDependencies.getPresentEmbeddingsInfo()
    }


@agent_router.post("/run")
def run_agent(request: AgentRequest):
    """
    Execute agent in specified mode.
    
    - **mode**: Agent operation mode (PLAN, TEACH, NOTES, RAG)
    - **topic**: Topic for TEACH/NOTES modes (optional)
    - **query**: Query for RAG mode or additional context (optional)
    
    Returns structured response based on mode:
    - PLAN: Topics, characters, quiz, and learning context
    - TEACH: Teaching content with examples and recap
    - NOTES: Structured notes with key points
    - RAG: Answer with sources and confidence
    """
    try:
        result =  run_agent_task(
            mode=request.mode,
            topic=request.topic,
            query=request.query
        )
        
        return {
            "mode": request.mode,
            "result": result,
            "summary": f"Agent executed in {request.mode} mode",
            "available_collections": avilable_collections
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")