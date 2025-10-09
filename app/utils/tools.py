## IMPORTS ##
from typing import Any, List, Dict
from ddgs import DDGS
import os
import json
from datetime import datetime
from langchain_core.tools import tool 
import chromadb
from app.services.rag_service import avilable_collections, RagPipeline

# Initialize RAG pipeline instance
rag_pipeline = RagPipeline()

## UTILITY FUNCTIONS

@tool
def format_results_for_llm(results: list) -> str:
    """
    Accepts a list of search result dicts and returns a single LLM-ready string.
    Format: [date] title\nbody\n\n

    Args:
        results (list): List of dictionaries with keys like 'title', 'body', 'href'

    Returns:
        str: Formatted string ready to pass into LLM
    """
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


## WEB SEARCH TOOLS

@tool
def searchWebTool(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    Useful for finding additional information not in the uploaded materials.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted search results as string
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        return format_results_for_llm(list(results))
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def searchNewsTool(query: str, max_results: int = 5) -> str:
    """
    Search news articles using DuckDuckGo.
    Useful for finding current events and recent developments on a topic.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted news results as string
    """
    try:
        results = DDGS().news(query, max_results=max_results)
        return format_results_for_llm(list(results))
    except Exception as e:
        return f"Error performing news search: {str(e)}"


## RAG TOOLS

@tool
def getPresentEmbeddingsInfo() -> str:
    """
    Get all the embeddings content description and information present in the RAG vector database.
    Useful to know which collection name has the right context to answer the query.
    
    Returns:
        Collection names with descriptions of their contents
    """
    result = []
    for key, value in avilable_collections.items():
        result.append(f"{key}: {value}\n")
        
    return "".join(result) if result else "No collections available in the database."


@tool
def createEmbeddingsFromText(
    text_content: str,
    collection_name: str,
    description: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_path: str = None
) -> str:
    """
    Create embeddings from text content and store in vector database.
    Use this when a user provides text content or when you need to save conversation context.
    
    Args:
        text_content: Text to create embeddings from
        collection_name: Name for the vector database collection
        description: Describe what the collection contains for easy identification
        chunk_size: Size of text chunks (default: 500)
        chunk_overlap: Overlap between chunks (default: 50)
        embedding_model: Name of the embedding model to use
        db_path: Path to store the database (None for in-memory)
        
    Returns:
        Success message with details
    """
    try:
        rag_pipeline.chunks_from_text(
            text_content=text_content,
            chunk_overlap=chunk_overlap
        )
        
        rag_pipeline.make_embeddings(
            embedding_model=embedding_model
        )
        
        rag_pipeline.save_embeddings(
            collection_name=collection_name,
            db_path=db_path
        )
        
        # Update available collections
        avilable_collections[collection_name] = description
        
        num_chunks = len(rag_pipeline.chunks)
        return f"Successfully created and stored {num_chunks} embeddings in collection '{collection_name}'"
    except Exception as e:
        return f"Error creating embeddings: {str(e)}"


@tool
def createEmbeddingsFromPDF(
    pdf_path: str,
    collection_name: str,
    description: str,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_path: str = None
) -> str:
    """
    Create embeddings from PDF file and store in vector database.
    Use this when a user uploads a PDF document.
    
    Args:
        pdf_path: Path to PDF file
        collection_name: Name for the vector database collection
        description: Describe what the collection contains
        chunk_overlap: Overlap between chunks (default: 50)
        embedding_model: Name of the embedding model to use
        db_path: Path to store the database (None for in-memory)
        
    Returns:
        Success message with details
    """
    try:
        rag_pipeline.chunks_from_pdf(
            pdf_path=pdf_path,
            chunk_overlap=chunk_overlap
        )
        
        rag_pipeline.make_embeddings(
            embedding_model=embedding_model
        )
        
        rag_pipeline.save_embeddings(
            collection_name=collection_name,
            db_path=db_path
        )
        
        # Update available collections
        avilable_collections[collection_name] = description
        
        num_chunks = len(rag_pipeline.chunks)
        num_pages = rag_pipeline.pages
        return f"Successfully processed {num_pages} pages and created {num_chunks} embeddings in collection '{collection_name}'"
    except Exception as e:
        return f"Error creating embeddings from PDF: {str(e)}"


@tool
def createEmbeddingsFromURL(
    pdf_url: str,
    collection_name: str,
    description: str,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_path: str = None
) -> str:
    """
    Create embeddings from PDF URL and store in vector database.
    Use this when a user provides a URL to a PDF document.
    
    Args:
        pdf_url: URL to PDF file
        collection_name: Name for the vector database collection
        description: Describe what the collection contains
        chunk_overlap: Overlap between chunks (default: 50)
        embedding_model: Name of the embedding model to use
        db_path: Path to store the database (None for in-memory)
        
    Returns:
        Success message with details
    """
    try:
        rag_pipeline.chunks_from_url(
            pdf_url=pdf_url,
            chunk_overlap=chunk_overlap
        )
        
        rag_pipeline.make_embeddings(
            embedding_model=embedding_model
        )
        
        rag_pipeline.save_embeddings(
            collection_name=collection_name,
            db_path=db_path
        )
        
        # Update available collections
        avilable_collections[collection_name] = description
        
        num_chunks = len(rag_pipeline.chunks)
        num_pages = rag_pipeline.pages
        return f"Successfully downloaded PDF and created {num_chunks} embeddings from {num_pages} pages in collection '{collection_name}'"
    except Exception as e:
        return f"Error creating embeddings from URL: {str(e)}"


@tool
def retrieveFromEmbeddings(
    collection_name: str,
    query: str,
    n_results: int = 5,
    db_path: str = None
) -> str:
    """
    Retrieve relevant information from embeddings using semantic search.
    This is the primary tool for accessing stored learning materials.
    
    Args:
        collection_name: Name of the collection to search
        query: Search query
        n_results: Number of results to return (default: 5)
        db_path: Path to the database
        
    Returns:
        Retrieved relevant text chunks
    """
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


@tool
def deleteEmbeddingsCollection(
    collection_name: str,
    db_path: str
) -> str:
    """
    Delete an embeddings collection from the database.
    Use with caution - this permanently removes data.
    
    Args:
        collection_name: Name of the collection to delete
        db_path: Path to the database
        
    Returns:
        Success message
    """
    try:
        RagPipeline.delete_data(collection_name, db_path)
        
        # Remove from available collections
        if collection_name in avilable_collections:
            del avilable_collections[collection_name]
            
        return f"Successfully deleted collection '{collection_name}'"
    except Exception as e:
        return f"Error deleting collection: {str(e)}"


## CONVERSATION MANAGEMENT TOOLS

@tool
def logConversation(user_message: str, ai_response: str) -> str:
    """
    Logs a user-AI conversation to a 'current_session' collection for long-term memory.
    This helps maintain context across the learning session.
    
    Args:
        user_message: The message from the user
        ai_response: The response from the AI
        
    Returns:
        Confirmation or error message
    """
    try:
        formatted_turn = f"User: {user_message}\nAI: {ai_response}\nTimestamp: {datetime.now().isoformat()}"
        
        createEmbeddingsFromText(
            text_content=formatted_turn,
            collection_name="current_session",
            description="Current learning session conversation history"
        )
        return "Successfully logged conversation turn to 'current_session'."
    except Exception as e:
        return f"Error logging conversation turn: {str(e)}"


@tool
def getConversationSummary(query: str, n_results: int = 10) -> str:
    """
    Retrieves relevant parts of the conversation history from the 'current_session' collection.
    Useful for understanding what has been discussed and maintaining context.
    
    Args:
        query: A query to find relevant parts of the conversation
        n_results: Number of conversation turns to retrieve (default: 10)
        
    Returns:
        Summary of relevant conversation history
    """
    try:
        retrieved_text = retrieveFromEmbeddings(
            collection_name="current_session",
            query=query,
            n_results=n_results
        )

        if "Error" in retrieved_text or "No relevant" in retrieved_text:
            return "No relevant conversation history found."

        # Extract content
        content_start = retrieved_text.find(":\n\n")
        if content_start != -1:
            return retrieved_text[content_start + 3:]
        
        return retrieved_text

    except Exception as e:
        return f"Error getting conversation summary: {str(e)}"


# TEXT PROCESSING TOOLS
@tool
def summarizeText(text: str, max_sentences: int = 5) -> str:
    """
    Summarize text to a specified number of sentences.
    Uses simple extractive summarization based on sentence importance.
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary (default: 5)
        
    Returns:
        Summarized text
    """
    try:
        # Split into sentences
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization - take first and last sentences plus middle ones
        if max_sentences >= 3:
            summary_sentences = [sentences[0]]  # First sentence
            
            # Add middle sentences
            step = len(sentences) // (max_sentences - 1)
            for i in range(1, max_sentences - 1):
                idx = min(i * step, len(sentences) - 2)
                summary_sentences.append(sentences[idx])
            
            summary_sentences.append(sentences[-1])  # Last sentence
        else:
            summary_sentences = sentences[:max_sentences]
        
        return '. '.join(summary_sentences) + '.'
    except Exception as e:
        return f"Error summarizing text: {str(e)}"


## UTILITY TOOLS
@tool
def getCurrentDateTime() -> str:
    """
    Get current date and time.
    Useful for timestamping and time-aware responses.
    
    Returns:
        Current date and time as formatted string
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def parseJSON(json_string: str) -> str:
    """
    Parse and validate JSON string.
    Useful for working with structured data.
    
    Args:
        json_string: JSON string to parse
        
    Returns:
        Formatted JSON or error message
    """
    try:
        parsed = json.loads(json_string)
        formatted = json.dumps(parsed, indent=2)
        return f"Valid JSON:\n{formatted}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


@tool
def calculateExpression(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    Useful for solving math problems in learning materials.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of calculation
    """
    try:
        # Only allow safe mathematical operations
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


## GAMIFICATION TOOLS

@tool
def generateQuizFromTopic(
    topic: str,
    collection_name: str,
    num_questions: int = 5,
    difficulty: str = "medium"
) -> str:
    """
    Generate quiz questions based on a specific topic from stored materials.
    
    Args:
        topic: The topic to generate quiz about
        collection_name: Collection containing the learning materials
        num_questions: Number of questions to generate (default: 5)
        difficulty: Difficulty level - easy, medium, or hard (default: medium)
        
    Returns:
        JSON string containing quiz questions
    """
    try:
        # Retrieve relevant content
        context = retrieveFromEmbeddings(
            collection_name=collection_name,
            query=topic,
            n_results=10
        )
        
        if "Error" in context:
            return context
        
        quiz_template = {
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "context": context[:1000],  # Limit context size
            "instructions": f"Generate {num_questions} {difficulty} multiple-choice questions about {topic}"
        }
        
        return json.dumps(quiz_template, indent=2)
    except Exception as e:
        return f"Error generating quiz: {str(e)}"


@tool
def trackLearningProgress(
    user_id: str,
    topic: str,
    score: float,
    time_spent: int,
    collection_name: str = "learning_progress"
) -> str:
    """
    Track user's learning progress for gamification.
    
    Args:
        user_id: Unique identifier for the user
        topic: Topic that was studied
        score: Score achieved (0-100)
        time_spent: Time spent in seconds
        collection_name: Collection to store progress (default: learning_progress)
        
    Returns:
        Confirmation message with progress details
    """
    try:
        progress_entry = {
            "user_id": user_id,
            "topic": topic,
            "score": score,
            "time_spent": time_spent,
            "timestamp": datetime.now().isoformat(),
            "status": "passed" if score >= 70 else "needs_review"
        }
        
        progress_text = json.dumps(progress_entry)
        
        createEmbeddingsFromText(
            text_content=progress_text,
            collection_name=collection_name,
            description="User learning progress and achievements"
        )
        
        return f"Progress tracked: {topic} - Score: {score}% - Status: {progress_entry['status']}"
    except Exception as e:
        return f"Error tracking progress: {str(e)}"


@tool
def getTopicDependencies(topic: str, collection_name: str) -> str:
    """
    Identify prerequisite topics that should be learned before the current topic.
    
    Args:
        topic: The topic to analyze
        collection_name: Collection containing learning materials
        
    Returns:
        List of prerequisite topics
    """
    try:
        # Retrieve context about the topic
        context = retrieveFromEmbeddings(
            collection_name=collection_name,
            query=f"prerequisites for {topic}",
            n_results=3
        )
        
        return f"Prerequisites and dependencies for {topic}:\n{context}"
    except Exception as e:
        return f"Error getting topic dependencies: {str(e)}"


## NPC CHARACTER TOOLS

@tool
def generateNPCPersonality(topic: str, teaching_style: str = "friendly") -> str:
    """
    Generate a personality profile for an NPC character that teaches a specific topic.
    
    Args:
        topic: The topic this NPC will teach
        teaching_style: Style of teaching (friendly, strict, humorous, etc.)
        
    Returns:
        JSON with NPC personality details
    """
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


@tool
def createLearningPath(
    topics: List[str],
    user_level: str = "beginner"
) -> str:
    """
    Create an optimized learning path through multiple topics.
    
    Args:
        topics: List of topics to learn
        user_level: User's current level (beginner, intermediate, advanced)
        
    Returns:
        Structured learning path with recommended order
    """
    try:
        learning_path = {
            "user_level": user_level,
            "total_topics": len(topics),
            "recommended_order": topics,
            "estimated_hours": len(topics) * 2,
            "milestones": [
                {"checkpoint": i + 1, "topic": topic}
                for i, topic in enumerate(topics)
            ]
        }
        
        return json.dumps(learning_path, indent=2)
    except Exception as e:
        return f"Error creating learning path: {str(e)}"


# EXPORT TOOLS LIST

# List of all available tools for the agent
tools = [
    # Web search
    searchWebTool,
    searchNewsTool,
    
    # RAG operations
    getPresentEmbeddingsInfo,
    createEmbeddingsFromText,
    createEmbeddingsFromPDF,
    createEmbeddingsFromURL,
    retrieveFromEmbeddings,
    deleteEmbeddingsCollection,
    
    # Conversation management
    logConversation,
    getConversationSummary,
    
    # Text processing
    summarizeText,
    
    # Utility
    getCurrentDateTime,
    parseJSON,
    calculateExpression,
    
    # Gamification
    generateQuizFromTopic,
    trackLearningProgress,
    getTopicDependencies,
    
    # NPC and learning path
    generateNPCPersonality,
    createLearningPath
]