## Imports
from fastapi import APIRouter, HTTPException
from app.schemas.response_schema import AgentRequest, AgentResponse
from app.services.agent_service import (
    run_agent_task,
    AgentMode,
    SupportDependencies,
    avilable_collections,
)

## Router instance
agent_router = APIRouter(prefix="/agent", tags=["AI Agent"])

@agent_router.get("/")
async def root():
    """Health check endpoint for the agent router."""
    return {"message": "Agent router is active."}

@agent_router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run AI agent in the selected mode.

    Modes:
    - PLAN  : Generate learning journey, topics, NPCs, quizzes
    - TEACH : Explain a topic interactively
    - NOTES : Generate summarized notes
    - RAG   : Retrieve and answer factual queries
    - TALK  : Chat conversationally 
    """
    try:
        mode = request.mode.lower()

        # PLAN mode - learning journey planner
        if mode == AgentMode.PLAN:
            state_result = await run_agent_task(
                mode='game',
                query=request.query
            )
            result: str = state_result  # run_agent_task now returns a string
            return AgentResponse(
                mode=AgentMode.PLAN,
                result=result,
                available_collections=avilable_collections
            )

        # TEACH mode - topic explanation
        elif mode == AgentMode.TEACH:
            if not request.topic:
                raise HTTPException(status_code=400, detail="Topic is required for TEACH mode.")
            state_result = await run_agent_task(
                mode='teach',
                topic=request.topic,
                query=request.query
            )
            result: str = state_result
            return AgentResponse(
                mode=AgentMode.TEACH,
                result=result,
                available_collections=avilable_collections
            )

        # NOTES mode - concise study notes
        elif mode == AgentMode.NOTES:
            if not request.topic:
                raise HTTPException(status_code=400, detail="Topic is required for NOTES mode.")
            state_result = await run_agent_task(
                mode='notes',
                topic=request.topic,
                query=request.query
            )
            result: str = state_result
            return AgentResponse(
                mode=AgentMode.NOTES,
                result=result,
                available_collections=avilable_collections
            )

        # RAG mode - retrieval-based Q&A
        elif mode == AgentMode.RAG:
            if not request.query:
                raise HTTPException(status_code=400, detail="Query is required for RAG mode.")
            state_result = await run_agent_task(
                mode='rag',
                query=request.query
            )
            result: str = state_result
            return AgentResponse(
                mode=AgentMode.RAG,
                result=result,
                available_collections=avilable_collections
            )
        # TALK mode - conversational chat
        elif mode == AgentMode.TALK:
            if not request.query:
                raise HTTPException(status_code=400, detail="Query is required for TALK mode.")
            state_result = await run_agent_task(
                mode='talk',
                query=request.query
            )
            result: str = state_result
            return AgentResponse(
                mode=AgentMode.TALK,
                result=result,
                available_collections=avilable_collections
            )

        # Invalid mode
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@agent_router.get("/collections")
async def get_available_collections():
    """Return all available RAG embedding collections."""
    try:
        if not avilable_collections:
            return {"message": "No collections available."}
        return {"available_collections": avilable_collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collections: {str(e)}")


@agent_router.get("/time")
async def get_current_time():
    """Return the current system date and time."""
    return {"datetime": SupportDependencies.getCurrentDateTime()}
