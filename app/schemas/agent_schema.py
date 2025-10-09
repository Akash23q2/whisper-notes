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
    

