## REQUEST/RESPONSE MODELS
from pydantic import BaseModel
from typing import List, Optional
from app.schemas.agent_schema import Quiz, GameCharacters
class SessionInitRequest(BaseModel):
    user_query: str
    user_id: Optional[str] = None


class MessageRequest(BaseModel):
    session_id: str
    message: str


class TopicSetRequest(BaseModel):
    session_id: str
    topic: str
    character: Optional[GameCharacters] = None


class QuizSubmissionRequest(BaseModel):
    session_id: str
    quiz_results: List[Quiz]


class UploadResourceRequest(BaseModel):
    collection_name: str
    description: str
    user_id: str


class WebSocketMessage(BaseModel):
    type: str  # 'chat', 'set_topic', 'generate_quiz', 'submit_quiz'
    session_id: str
    data: dict

class AgentRequest(BaseModel):
    mode: str
    topic: Optional[str] = None
    query: Optional[str] = None

class AgentResponse(BaseModel):
    mode: str
    result: str
    summary: Optional[str] = None
    available_collections: Optional[dict] = None