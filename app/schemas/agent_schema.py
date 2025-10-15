from pydantic import BaseModel
from typing import List, Optional

#doc string definations for agent understanding of context

class Quiz(BaseModel):
    """Quiz questions"""
    question:str
    """list of options"""
    mcq_options:list[str]
    """correct answer"""
    correct_ans:str
    """explanation of answer"""
    description:str #explanation of answer
    
class GameCharacters(BaseModel):
    """character name"""
    name:str
    """topic to teach or test"""
    topic:str
    """whether to assist or test on that topic"""
    description:str #wheter it wants to assist or test on that topic
    
class Topics(BaseModel):
    """topics extracted from the learning content"""
    topic:str
    """description of the topic"""
    description:str
    

## AgentMode
class AgentMode(str):
    """Defines operational modes of the agent."""
    PLAN = "plan_learning_journey"
    TEACH = "teach_topic"
    NOTES = "generate_notes"
    RAG = "rag_query"
    TALK = "talk"
    
## Agent output schemas for different modes ##
from pydantic import BaseModel
from typing import List, Dict

## Base state for generic RAG responses ##
class AgentState(BaseModel):
    output: str

class SummaryState(BaseModel):
    summary: str

## Notes generator output ##
class NotesState(BaseModel):
    topic: str
    notes: str  # summarized notes
    bullet_points: List[str] = []  # optional structured points

## Teaching mode output ##
class TeachState(BaseModel):
    topic: str
    explanation: str
    examples: List[str] = []  # optional examples or analogies
    self_check_questions: List[Dict[str, str]] = []  # list of question dicts: {"question":..., "answer":...}

## Game mode / quiz / NPC generation ##
class GameState(BaseModel):
    topics: List[str] = []
    characters: List[Dict[str, str]] = []  # NPC characters and personalities
    quiz: List[Dict[str, str]] = []        # multiple-choice questions with answers
    summary: str = ""  # short recap or description
