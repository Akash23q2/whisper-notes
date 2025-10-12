## Imports ##
from typing import TypedDict, Annotated, Sequence, Literal, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from datetime import datetime
from app.schemas.agent_schema import AgentState, NotesState, TeachState, GameState, AgentMode
from app.services.rag_service import avilable_collections, RagPipeline, data_injestion
from ddgs import DDGS
import chromadb
import os
import asyncio
from dotenv import load_dotenv
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

## Initialize RAG Pipeline ##
rag_pipeline = RagPipeline()

## Initialize AI agent ##
from pydantic_ai.models.google import GoogleModel

model = GoogleModel('gemini-2.5-flash')
agent = Agent(model)

## Dependencies ##
@dataclass
class SupportDependencies:
    ## RAG Dependencies ##
    def getPresentEmbeddingsInfo() -> str:
        result = []
        for key, value in avilable_collections.items():
            result.append(f"{key}: {value}\n")
        return "".join(result) if result else "No collections available in the database."

    def getConversationSummary(query: str, n_results: int = 10) -> str:
        try:
            retrieved_text = retrieveFromEmbeddings(
                collection_name="current_session",
                query=query,
                n_results=n_results
            )
            if "Error" in retrieved_text or "No relevant" in retrieved_text:
                return "No relevant conversation history found."
            content_start = retrieved_text.find(":\n\n")
            if content_start != -1:
                return retrieved_text[content_start + 3:]
            return retrieved_text
        except Exception as e:
            return f"Error getting conversation summary: {str(e)}"

    def getCurrentDateTime() -> str:
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

## Utility Functions ##
def format_results_for_llm(results: list) -> str:
    if not results:
        return "No search results found."
    lines = []
    for item in results:
        title = item.get('title', '').strip()
        body = item.get('body', item.get('snippet', '')).strip()
        date = item.get('date', 'N/A').strip()
        if title and body:
            formatted = f"[{date}] {title}\n{body}\n"
            lines.append(formatted)
    return "\n".join(lines) if lines else "No valid results found."

## Tools Definitions ##
@agent.tool
def queryAllEmbeddings(ctx: RunContext[SupportDependencies], query: str, n_results: int = 5) -> str:
    """
    Search all available embeddings, prioritizing current_session for memory recall.
    If nothing relevant, fallback to web search.
    """
    try:
        # Always try 'current_session' first
        if 'current_session' in avilable_collections:
            result = retrieveFromEmbeddings(ctx, collection_name='current_session', query=query, n_results=n_results)
            if result and "No relevant" not in result and "Error" not in result:
                return f"Answer from 'current_session':\n{result}"

        # Then try all other embeddings
        for collection_name in avilable_collections.keys():
            if collection_name == 'current_session':
                continue
            result = retrieveFromEmbeddings(ctx, collection_name=collection_name, query=query, n_results=n_results)
            if result and "No relevant" not in result and "Error" not in result:
                return f"Answer from '{collection_name}' collection:\n{result}"
        
        # fallback to web search
        web_result = searchWebTool(ctx, query=query, max_results=n_results)
        return f"No relevant embedding found. Fallback to web search:\n{web_result}"
    except Exception as e:
        return f"Error querying embeddings: {str(e)}"

@agent.tool
def retrieveFromEmbeddings(
    ctx: RunContext[SupportDependencies],
    collection_name: str,
    query: str,
    n_results: int = 5,
    db_path: str = None
) -> str:
    try:
        if db_path:
            rag_pipeline.client = chromadb.PersistentClient(path=db_path)
        results = rag_pipeline.retrieve(
            collection_name=collection_name,
            query=query,
            n_results=n_results
        )
        formatted_results = "\n\n---\n\n".join(results)
        return f"Retrieved {len(results)} relevant chunks:\n\n{formatted_results}"
    except Exception as e:
        return f"Error retrieving from embeddings: {str(e)}"

@agent.tool
def searchWebTool(ctx: RunContext[SupportDependencies], query: str, max_results: int = 5) -> str:
    try:
        results = DDGS().text(query, max_results=max_results)
        return format_results_for_llm(list(results))
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@agent.tool
def logConversation(ctx: RunContext[SupportDependencies], user_message: str, ai_response: str) -> str:
    try:
        formatted_turn = f"User: {user_message}\nAI: {ai_response}\nTimestamp: {datetime.now().isoformat()}"
        data_injestion(
            text_content=formatted_turn,
            collection_name="current_session",
            description="Current learning session conversation history"
        )
        return "Successfully logged conversation turn to 'current_session'."
    except Exception as e:
        return f"Error logging conversation turn: {str(e)}"

@agent.tool
def calculateExpression(ctx: RunContext[SupportDependencies], expression: str) -> str:
    try:
        allowed_names = {'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

## Gamification Tools ##
@agent.tool
def generateQuizFromTopic(
    ctx: RunContext[SupportDependencies],
    topic: str,
    collection_name: str,
    num_questions: int = 5,
    difficulty: str = "medium"
) -> str:
    try:
        context = retrieveFromEmbeddings(ctx, collection_name=collection_name, query=topic, n_results=10)
        if "Error" in context:
            return context
        quiz_template = {
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "context": context[:1000],
            "instructions": f"Generate {num_questions} {difficulty} multiple-choice questions about {topic}"
        }
        return json.dumps(quiz_template, indent=2)
    except Exception as e:
        return f"Error generating quiz: {str(e)}"

## NPC Character Tools ##
@agent.tool
def generateNPCPersonality(ctx: RunContext[SupportDependencies], topic: str, teaching_style: str = "friendly") -> str:
    try:
        personality = {
            "topic": topic,
            "teaching_style": teaching_style,
            "name_suggestions": [
                f"Professor {topic.split()[0] if topic.split() else 'Learning'}",
                f"{topic} Master",
                f"Guide {topic[:10]}"
            ],
            "personality_traits": {
                "friendly": ["encouraging", "patient", "uses analogies"],
                "strict": ["direct", "demanding", "high standards"],
                "humorous": ["jokes", "funny examples", "light-hearted"]
            }.get(teaching_style, ["balanced", "clear", "helpful"]),
            "greeting_template": f"Hello! I'm here to help you master {topic}!",
            "teaching_approach": f"I specialize in {topic} and love to teach using {teaching_style} methods."
        }
        return json.dumps(personality, indent=2)
    except Exception as e:
        return f"Error generating NPC personality: {str(e)}"

## Prompts ##

def get_memory_snippet(query_context: str = "", n_results: int = 10) -> str:
    """
    Retrieves relevant context from current_session embeddings.
    Returns empty string if nothing is found.
    """
    memory = SupportDependencies.getConversationSummary(query=query_context, n_results=n_results)
    if "No relevant" in memory or "Error" in memory:
        return ""
    return f"Relevant past conversation:\n{memory}\n"


base_context = f"""
You are an intelligent educational assistant with access to a Retrieval-Augmented Generation (RAG) system.
You can retrieve and reason over the following available collections:
{SupportDependencies.getPresentEmbeddingsInfo()}

Use `retrieveFromEmbeddings` tool whenever you need more information.
At the end, always use `logConversation` to record the session summary.
"""

def game_mode_prompt(query_context: str = ""):
    memory = get_memory_snippet(query_context)
    return f"""{base_context}
{memory}
### Role: Learning Journey Planner
Tasks:
1. Analyze available RAG content.
2. Extract key topics.
3. Create engaging NPC characters and quiz questions.
4. Return structured data with topics, characters, quiz, summary.
Create a quiz with 5 multiple-choice questions:
- Test understanding of key concepts
- Range from basic recall to application
- Include 4 options each (A, B, C, D)
- Clear, unambiguous correct answers with explanations
Be creative, fun, and clear.
{query_context}
"""

def teach_topic_prompt(query_context: str = "", topic: str = "filters"):
    memory = get_memory_snippet(query_context)
    return f"""{base_context}
{memory}
### Role: Interactive Teacher
Explain the topic '{topic}' in an engaging way.
- Use retrieved RAG context and past conversation memory for examples.
- Maintain personality (calm, excited, etc.).
- Step-by-step teaching with short explanations.
- End with a short recap and 5 self-check questions.
{query_context}
"""

def notes_generate_prompt(query_context: str = "", topic: str = "filters"):
    memory = get_memory_snippet(query_context)
    return f"""{base_context}
{memory}
### Role: Notes Generator
- Retrieve content related to '{topic}'.
- Include relevant past conversation context.
- Summarize into concise, easy-to-understand notes.
- Prefer structured bullet points or markdown sections.
{query_context}
"""

def rag_query_prompt(query_context: str = ""):
    memory = get_memory_snippet(query_context)
    return f"""{base_context}
{memory}
Retrieve and answer any factual or conceptual question:
- First, check the `current_session` embedding for past conversation context.
- Then, use the `queryAllEmbeddings` tool to find relevant info from all other embeddings.
- If no relevant info is found in embeddings, use `searchWebTool`.
- Always log the interaction using `logConversation`.
- Cite the relevant embedding collection in your answer.
{query_context}
"""



## Unified Agent Run Function ##
def run_agent_task(
    mode: Literal['game', 'teach', 'notes', 'rag'],
    query: str = "",
    topic: str = "",
    agent: Agent = agent
):
    ## Run agent based on mode ##
    if mode == 'game':
        result = agent.run_sync(
            game_mode_prompt(query_context=query),
            deps=SupportDependencies,
            output_type=GameState
        )
    elif mode == 'teach':
        result = agent.run_sync(
            teach_topic_prompt(query_context=query, topic=topic),
            deps=SupportDependencies,
            output_type=TeachState
        )
    elif mode == 'notes':
        result = agent.run_sync(
            notes_generate_prompt(query_context=query, topic=topic),
            deps=SupportDependencies,
            output_type=NotesState
        )
    elif mode == 'rag':
        result = agent.run_sync(
            rag_query_prompt(query_context=query),
            deps=SupportDependencies,
            output_type=AgentState
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(result.output)
    return result.output
