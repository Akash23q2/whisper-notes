## imports
from app.schemas.response_schema import SessionInitRequest, MessageRequest, TopicSetRequest, QuizSubmissionRequest, UploadResourceRequest, WebSocketMessage
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import asyncio
import json
from app.services.agent_service import LearningAgentService
from app.schemas.agent_schema import QuizResults, GameCharacters, Topics
import os

app = FastAPI(title="Gamified Learning Platform API")

# Add CORS middleware for WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions and WebSocket connections
active_sessions: Dict[str, dict] = {}
active_websockets: Dict[str, WebSocket] = {}



## WEBSOCKET CONNECTION MANAGER

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

    async def stream_message(self, session_id: str, content: str, chunk_size: int = 10):
        """Stream message in chunks for typing effect"""
        if session_id in self.active_connections:
            words = content.split()
            current_chunk = []
            
            for i, word in enumerate(words):
                current_chunk.append(word)
                
                if len(current_chunk) >= chunk_size or i == len(words) - 1:
                    chunk_text = ' '.join(current_chunk)
                    await self.active_connections[session_id].send_json({
                        "type": "stream_chunk",
                        "content": chunk_text,
                        "is_final": i == len(words) - 1
                    })
                    current_chunk = []
                    await asyncio.sleep(0.05)  # Small delay for natural feel


manager = ConnectionManager()


## WEBSOCKET ENDPOINT

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time communication
    Handles chat, quiz generation, and live updates
    """
    await manager.connect(session_id, websocket)
    
    try:
        # Send connection confirmation
        await manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "message": "WebSocket connection established"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if session_id not in active_sessions:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": "Session not found. Please create a session first."
                })
                continue
            
            agent_service = active_sessions[session_id]["agent"]
            
            # Handle different message types
            if message_type == "chat":
                await handle_chat_message(session_id, agent_service, data.get("message", ""))
                
            elif message_type == "set_topic":
                await handle_set_topic(session_id, agent_service, data)
                
            elif message_type == "generate_quiz":
                await handle_generate_quiz(session_id, agent_service, data.get("topic", ""))
                
            elif message_type == "submit_quiz":
                await handle_submit_quiz(session_id, agent_service, data.get("quiz_results", []))
                
            elif message_type == "get_progress":
                await handle_get_progress(session_id, agent_service)
                
            elif message_type == "ping":
                await manager.send_message(session_id, {"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(session_id)


## WEBSOCKET MESSAGE HANDLERS

async def handle_chat_message(session_id: str, agent_service: LearningAgentService, message: str):
    """Handle chat messages with streaming response"""
    try:
        # Send typing indicator
        await manager.send_message(session_id, {
            "type": "typing",
            "is_typing": True
        })
        
        # Process message
        result = agent_service.process_message(message)
        
        # Get AI response
        ai_response = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                ai_response = msg.content
                break
        
        # Stop typing indicator
        await manager.send_message(session_id, {
            "type": "typing",
            "is_typing": False
        })
        
        # Stream the response
        await manager.stream_message(session_id, ai_response)
        
        # Send complete response with metadata
        await manager.send_message(session_id, {
            "type": "chat_response",
            "content": ai_response,
            "next_step": result.get("next_step", ""),
            "current_topic": result.get("current_topic", ""),
            "timestamp": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error processing chat: {str(e)}"
        })


async def handle_set_topic(session_id: str, agent_service: LearningAgentService, data: dict):
    """Handle topic setting with immediate feedback"""
    try:
        topic = data.get("topic", "")
        character_data = data.get("character")
        character = GameCharacters(**character_data) if character_data else None
        
        # Send acknowledgment
        await manager.send_message(session_id, {
            "type": "topic_setting",
            "status": "processing",
            "topic": topic
        })
        
        # Set topic
        agent_service.set_current_topic(topic, character)
        
        # Get initial teaching response
        result = agent_service.process_message(f"Let's learn about {topic}")
        
        # Get response
        ai_response = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                ai_response = msg.content
                break
        
        # Stream the teaching introduction
        await manager.stream_message(session_id, ai_response)
        
        # Send complete topic set confirmation
        await manager.send_message(session_id, {
            "type": "topic_set",
            "topic": topic,
            "character": character.dict() if character else None,
            "teaching_response": ai_response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error setting topic: {str(e)}"
        })


async def handle_generate_quiz(session_id: str, agent_service: LearningAgentService, topic: str):
    """Handle quiz generation with progress updates"""
    try:
        # Send generation started
        await manager.send_message(session_id, {
            "type": "quiz_generating",
            "status": "started",
            "topic": topic
        })
        
        # Generate quiz
        result = agent_service.process_message("Generate a quiz for me to test my understanding")
        
        # Extract quiz
        ai_response = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                ai_response = msg.content
                break
        
        # Send progress update
        await manager.send_message(session_id, {
            "type": "quiz_generating",
            "status": "parsing",
            "progress": 50
        })
        
        # Parse quiz questions (attempt to extract JSON)
        try:
            # Try to find JSON in response
            quiz_data = json.loads(ai_response) if ai_response.startswith('[') else None
        except:
            quiz_data = None
        
        # Send complete quiz
        await manager.send_message(session_id, {
            "type": "quiz_generated",
            "topic": topic,
            "quiz": ai_response,
            "parsed_questions": quiz_data,
            "timestamp": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error generating quiz: {str(e)}"
        })


async def handle_submit_quiz(session_id: str, agent_service: LearningAgentService, quiz_results: list):
    """Handle quiz submission with real-time evaluation"""
    try:
        # Convert to QuizResults objects
        quiz_results_objs = [QuizResults(**q) for q in quiz_results]
        
        # Send evaluation started
        await manager.send_message(session_id, {
            "type": "quiz_evaluating",
            "status": "started"
        })
        
        # Calculate score
        total = len(quiz_results_objs)
        correct = sum(1 for q in quiz_results_objs if q.user_ans.upper() == q.correct_ans.upper())
        score = (correct / total * 100) if total > 0 else 0
        passed = score >= 70
        
        # Send quick score update
        await manager.send_message(session_id, {
            "type": "quiz_score",
            "score": score,
            "correct_answers": correct,
            "total_questions": total,
            "passed": passed
        })
        
        # Get detailed evaluation
        result = agent_service.submit_quiz_answers(quiz_results_objs)
        
        # Get evaluation response
        evaluation = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                evaluation = msg.content
                break
        
        # Stream evaluation
        await manager.stream_message(session_id, evaluation)
        
        # Send complete evaluation
        await manager.send_message(session_id, {
            "type": "quiz_evaluated",
            "score": score,
            "correct_answers": correct,
            "total_questions": total,
            "passed": passed,
            "evaluation": evaluation,
            "next_step": result.get("next_step", ""),
            "timestamp": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error evaluating quiz: {str(e)}"
        })


async def handle_get_progress(session_id: str, agent_service: LearningAgentService):
    """Get real-time progress update"""
    try:
        progress = agent_service.get_progress()
        
        # Calculate statistics
        total_topics = len(progress)
        passed_topics = sum(1 for p in progress.values() if p.get("passed", False))
        avg_score = sum(p.get("score", 0) for p in progress.values()) / total_topics if total_topics > 0 else 0
        
        await manager.send_message(session_id, {
            "type": "progress_update",
            "total_topics": total_topics,
            "passed_topics": passed_topics,
            "average_score": avg_score,
            "completion_rate": (passed_topics / total_topics * 100) if total_topics > 0 else 0,
            "detailed_progress": progress,
            "timestamp": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Error getting progress: {str(e)}"
        })


## REST API ENDPOINTS (for non-real-time operations)

@app.post("/api/v1/session/create")
async def create_session(request: SessionInitRequest):
    """Create a new learning session"""
    try:
        session_id = str(uuid.uuid4())
        agent_service = LearningAgentService()
        initial_state = agent_service.initialize_session(request.user_query)
        
        active_sessions[session_id] = {
            "agent": agent_service,
            "user_id": request.user_id,
            "created_at": asyncio.get_event_loop().time()
        }
        
        return {
            "session_id": session_id,
            "status": "created",
            "message": "Learning session initialized successfully",
            "initial_response": initial_state["messages"][-1].content if initial_state["messages"] else "",
            "websocket_url": f"/ws/{session_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent_service = active_sessions[session_id]["agent"]
        state = agent_service.current_state
        
        return {
            "session_id": session_id,
            "current_topic": state.get("current_topic", ""),
            "topics": state.get("topics", []),
            "characters": state.get("characters", []),
            "progress": agent_service.get_progress(),
            "next_step": state.get("next_step", ""),
            "websocket_connected": session_id in manager.active_connections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/session/{session_id}")
async def end_session(session_id: str):
    """End a learning session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent_service = active_sessions[session_id]["agent"]
        summary = agent_service.end_session()
        
        # Notify via WebSocket if connected
        if session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "session_ended",
                "summary": summary
            })
            manager.disconnect(session_id)
        
        del active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": "ended",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


## RESOURCE UPLOAD ENDPOINTS (REST - file uploads)
@app.post("/api/v1/resources/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    description: str = Form(...),
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Upload PDF with real-time progress updates"""
    try:
        # Notify via WebSocket if connected
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_progress",
                "status": "uploading",
                "filename": file.filename,
                "progress": 0
            })
        
        # Save file
        file_path = f"/tmp/{user_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_progress",
                "status": "processing",
                "progress": 50
            })
        
        # Create embeddings
        from app.utils.tools import createEmbeddingsFromPDF
        result = createEmbeddingsFromPDF.invoke({
            "pdf_path": file_path,
            "collection_name": collection_name,
            "description": description
        })
        
        os.remove(file_path)
        
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_complete",
                "status": "completed",
                "collection_name": collection_name,
                "message": result,
                "progress": 100
            })
        
        return {
            "status": "success",
            "message": result,
            "collection_name": collection_name
        }
    except Exception as e:
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_error",
                "error": str(e)
            })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/resources/upload/text")
async def upload_text(
    text_content: str = Form(...),
    collection_name: str = Form(...),
    description: str = Form(...),
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Upload text content with real-time updates"""
    try:
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_progress",
                "status": "processing",
                "progress": 50
            })
        
        from app.utils.tools import createEmbeddingsFromText
        result = createEmbeddingsFromText.invoke({
            "text_content": text_content,
            "collection_name": collection_name,
            "description": description
        })
        
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_complete",
                "status": "completed",
                "collection_name": collection_name,
                "message": result,
                "progress": 100
            })
        
        return {
            "status": "success",
            "message": result,
            "collection_name": collection_name
        }
    except Exception as e:
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_error",
                "error": str(e)
            })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/resources/upload/url")
async def upload_url(
    pdf_url: str = Form(...),
    collection_name: str = Form(...),
    description: str = Form(...),
    user_id: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Upload PDF from URL with progress tracking"""
    try:
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_progress",
                "status": "downloading",
                "progress": 25
            })
        
        from app.utils.tools import createEmbeddingsFromURL
        result = createEmbeddingsFromURL.invoke({
            "pdf_url": pdf_url,
            "collection_name": collection_name,
            "description": description
        })
        
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_complete",
                "status": "completed",
                "collection_name": collection_name,
                "message": result,
                "progress": 100
            })
        
        return {
            "status": "success",
            "message": result,
            "collection_name": collection_name
        }
    except Exception as e:
        if session_id and session_id in manager.active_connections:
            await manager.send_message(session_id, {
                "type": "upload_error",
                "error": str(e)
            })
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/resources/collections")
async def get_collections():
    """Get all available collections"""
    try:
        from app.utils.tools import getPresentEmbeddingsInfo
        collections = getPresentEmbeddingsInfo.invoke({})
        
        return {
            "status": "success",
            "collections": collections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """Health check with connection statistics"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "active_websockets": len(manager.active_connections)
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)