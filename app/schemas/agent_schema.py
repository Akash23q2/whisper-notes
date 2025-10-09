from pydantic import BaseModel
from typing import List, Optional

class QuizResults(BaseModel):
    question:str
    mcq_options:list[str]
    correct_ans:str
    user_ans:str
    
class GameCharacters(BaseModel):
    name:str
    topic:str
    description:str #wheter it wants to assist or test on that topic
    
class Topics(BaseModel):
    topic:str
    description:str
    
# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

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
    quiz_results: List[QuizResults]


class UploadResourceRequest(BaseModel):
    collection_name: str
    description: str
    user_id: str


class WebSocketMessage(BaseModel):
    type: str  # 'chat', 'set_topic', 'generate_quiz', 'submit_quiz'
    session_id: str
    data: dict

