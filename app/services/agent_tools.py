##imports##
from typing import Any, List, TypedDict, Literal, Dict
from ddgs import DDGS
import os
import json
from datetime import datetime
from langchain_core.tools import tool 

# Import RAG service
from app.services.rag_service import RagPipeline

# Initialize RAG pipeline instance
rag_pipeline = RagPipeline()

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
        # Handle different possible key names from DDGS
        title = item.get('title', '').strip()
        body = item.get('body', item.get('snippet', '')).strip()
        date = item.get('date', 'N/A').strip()
        
        if title and body:
            formatted = f"[{date}] {title}\n{body}\n"
            lines.append(formatted)
    
    return "\n".join(lines) if lines else "No valid results found."


## WEB SEARCH TOOLS ##
@tool
def searchWebTool(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
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
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted news results as string
    """
    try:
        results = DDGS().news(query, max_results=max_results)
        return format_results_for_llm(list(results))
    except Exception as e:
        return f"Error performing news search: {str(e)}"


## RAG TOOLS ##
@tool
def createEmbeddingsFromText(
    text_content: str,
    collection_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_path: str = None
) -> str:
    """
    Create embeddings from text content and store in vector database.
    
    Args:
        text_content: Text to create embeddings from
        collection_name: Name for the vector database collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Name of the embedding model to use
        db_path: Path to store the database (None for in-memory)
        
    Returns:
        Success message with details
    """
    try:
        # Create chunks from text
        rag_pipeline.chunks_from_text(
            text_content=text_content,
            chunk_overlap=chunk_overlap
        )
        
        # Generate embeddings
        rag_pipeline.make_embeddings(
            embedding_model=embedding_model
        )
        
        # Save to vector database
        rag_pipeline.save_embeddings(
            collection_name=collection_name,
            db_path=db_path
        )
        
        num_chunks = len(rag_pipeline.chunks)
        return f"Successfully created and stored {num_chunks} embeddings in collection '{collection_name}'"
    except Exception as e:
        return f"Error creating embeddings: {str(e)}"

@tool
def createEmbeddingsFromPDF(
    pdf_path: str,
    collection_name: str,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_path: str = None
) -> str:
    """
    Create embeddings from PDF file and store in vector database.
    
    Args:
        pdf_path: Path to PDF file
        collection_name: Name for the vector database collection
        chunk_overlap: Overlap between chunks
        embedding_model: Name of the embedding model to use
        db_path: Path to store the database (None for in-memory)
        
    Returns:
        Success message with details
    """
    try:
        # Create chunks from PDF
        rag_pipeline.chunks_from_pdf(
            pdf_path=pdf_path,
            chunk_overlap=chunk_overlap
        )
        
        # Generate embeddings
        rag_pipeline.make_embeddings(
            embedding_model=embedding_model
        )
        
        # Save to vector database
        rag_pipeline.save_embeddings(
            collection_name=collection_name,
            db_path=db_path
        )
        
        num_chunks = len(rag_pipeline.chunks)
        num_pages = rag_pipeline.pages
        return f"Successfully processed {num_pages} pages and created {num_chunks} embeddings in collection '{collection_name}'"
    except Exception as e:
        return f"Error creating embeddings from PDF: {str(e)}"

@tool
def createEmbeddingsFromURL(
    pdf_url: str,
    collection_name: str,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_path: str = None
) -> str:
    """
    Create embeddings from PDF URL and store in vector database.
    
    Args:
        pdf_url: URL to PDF file
        collection_name: Name for the vector database collection
        chunk_overlap: Overlap between chunks
        embedding_model: Name of the embedding model to use
        db_path: Path to store the database (None for in-memory)
        
    Returns:
        Success message with details
    """
    try:
        # Create chunks from PDF URL
        rag_pipeline.chunks_from_url(
            pdf_url=pdf_url,
            chunk_overlap=chunk_overlap
        )
        
        # Generate embeddings
        rag_pipeline.make_embeddings(
            embedding_model=embedding_model
        )
        
        # Save to vector database
        rag_pipeline.save_embeddings(
            collection_name=collection_name,
            db_path=db_path
        )
        
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
    
    Args:
        collection_name: Name of the collection to search
        query: Search query
        n_results: Number of results to return
        db_path: Path to the database
        
    Returns:
        Retrieved relevant text chunks
    """
    try:
        # Initialize client if needed
        if db_path and not rag_pipeline.client:
            import chromadb
            rag_pipeline.client = chromadb.PersistentClient(path=db_path)
        elif not rag_pipeline.client:
            import chromadb
            rag_pipeline.client = chromadb.Client()
        
        # Retrieve results
        results = rag_pipeline.retrieve(
            collection_name=collection_name,
            query=query,
            n_results=n_results
        )
        
        # Format results
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
    
    Args:
        collection_name: Name of the collection to delete
        db_path: Path to the database
        
    Returns:
        Success message
    """
    try:
        RagPipeline.delete_data(collection_name, db_path)
        return f"Successfully deleted collection '{collection_name}'"
    except Exception as e:
        return f"Error deleting collection: {str(e)}"


## UTILITY TOOLS ##
@tool
def readFile(file_path: str) -> str:
    """
    Read contents of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File contents as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File '{file_path}' contents:\n\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def writeFile(file_path: str, content: str, mode: str = 'w') -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        mode: Write mode ('w' for write, 'a' for append)
        
    Returns:
        Success message
    """
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def listFiles(directory_path: str, extension_filter: str = None) -> str:
    """
    List files in a directory.
    
    Args:
        directory_path: Path to directory
        extension_filter: Optional file extension filter (e.g., '.txt', '.pdf')
        
    Returns:
        List of files as string
    """
    try:
        if not os.path.exists(directory_path):
            return f"Directory '{directory_path}' does not exist"
        
        files = os.listdir(directory_path)
        
        if extension_filter:
            files = [f for f in files if f.endswith(extension_filter)]
        
        if not files:
            return f"No files found in '{directory_path}'"
        
        file_list = "\n".join(files)
        return f"Files in '{directory_path}':\n{file_list}"
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool
def getCurrentDateTime() -> str:
    """
    Get current date and time.
    
    Returns:
        Current date and time as formatted string
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def parseJSON(json_string: str) -> str:
    """
    Parse and validate JSON string.
    
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

## TEXT SUMMARIZATION TOOLS ##
@tool
def logConversationTurn(user_message: str, ai_response: str) -> str:
    """
    Logs a user-AI conversation turn to a 'current_session' collection for long-term memory.
    This is useful for remembering the context of the current conversation.
    
    Args:
        user_message: The message from the user.
        ai_response: The response from the AI.
        
    Returns:
        A confirmation or error message.
    """
    try:
        formatted_turn = f"user: {user_message}\nai: {ai_response}"
        
        # Call the existing embedding creation function to append the turn to the session.
        createEmbeddingsFromText(
            text_content=formatted_turn,
            collection_name="current_session"
        )
        return "Successfully logged conversation turn to 'current_session'."
    except Exception as e:
        return f"Error logging conversation turn: {str(e)}"

@tool
def summarizeText(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple extractive summary of text.
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summarized text
    """
    try:
        # Split into sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Take first and last sentences, and one from middle
        summary_sentences = []
        summary_sentences.append(sentences[0])
        if max_sentences > 2:
            summary_sentences.append(sentences[len(sentences)//2])
        summary_sentences.append(sentences[-1])
        
        return ". ".join(summary_sentences[:max_sentences]) + "."
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

@tool
def getConversationSummary(query: str, n_results: int = 10, max_sentences: int = 5) -> str:
    """
    Retrieves and summarizes the conversation history from the 'current_session' collection.
    This is useful for getting a summary of the ongoing conversation to understand context.
    
    Args:
        query (str): A query to find relevant parts of the conversation. Use a general query like "conversation summary" to get an overview.
        n_results (int): The number of conversation turns to retrieve for context, if not enough , try to increase this value.
        max_sentences (int): The maximum number of sentences for the final summary.
        
    Returns:
        A summary of the conversation history.
    """
    try:
        # Step 1: Retrieve conversation history
        retrieved_text_with_header = retrieveFromEmbeddings(
            collection_name="current_session",
            query=query,
            n_results=n_results
        )

        if "Error" in retrieved_text_with_header:
            return f"Could not retrieve conversation history: {retrieved_text_with_header}"

        # Extract the actual content from the formatted string returned by retrieveFromEmbeddings
        content_start_index = retrieved_text_with_header.find(":\n\n")
        if content_start_index == -1 or "No relevant chunks" in retrieved_text_with_header:
            return "No relevant conversation history found to summarize."
            
        actual_content = retrieved_text_with_header[content_start_index + 3:]

        # Step 2: Summarize the retrieved text
        summary = summarizeText(text=actual_content, max_sentences=max_sentences)

        return f"Summary of conversation context:\n{summary}"

    except Exception as e:
        return f"An error occurred while getting the conversation summary: {str(e)}"

#enhanced tools 
@tool
def extractTopicsFromText(text: str) -> str:
    """
    Extract main topics and concepts from text.
    
    Args:
        text: Text to extract topics from
        
    Returns:
        JSON string with extracted topics
    """
    try:
        # Simple keyword extraction
        lines = text.split('\n')
        topics = []
        
        keywords = ['about', 'learn', 'teach', 'explain', 'topic', 'subject']
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                # Extract the part after the keyword
                for kw in keywords:
                    if kw in line_lower:
                        parts = line_lower.split(kw)
                        if len(parts) > 1:
                            topic = parts[1].strip().split('.')[0].strip(',')
                            if topic and len(topic) > 3:
                                topics.append(topic)
        
        # Remove duplicates
        topics = list(set(topics))
        
        result = {
            "topics": topics[:10],  # Limit to top 10
            "count": len(topics)
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error extracting topics: {str(e)}"

@tool
def createLearningPath(topics: List[str], difficulty: str = "beginner") -> str:
    """
    Create a structured learning path from topics.
    
    Args:
        topics: List of topics to create path from
        difficulty: Difficulty level (beginner/intermediate/advanced)
        
    Returns:
        Structured learning path as JSON
    """
    try:
        if isinstance(topics, str):
            topics = json.loads(topics) if topics.startswith('[') else [topics]
        
        # Simple ordering: fundamentals first
        fundamental_keywords = ['basic', 'intro', 'fundamental', 'overview']
        advanced_keywords = ['advanced', 'deep', 'complex', 'optimization']
        
        fundamentals = []
        intermediate = []
        advanced = []
        
        for topic in topics:
            topic_lower = topic.lower()
            if any(kw in topic_lower for kw in fundamental_keywords):
                fundamentals.append(topic)
            elif any(kw in topic_lower for kw in advanced_keywords):
                advanced.append(topic)
            else:
                intermediate.append(topic)
        
        ordered_path = fundamentals + intermediate + advanced
        
        path = {
            "difficulty": difficulty,
            "total_topics": len(ordered_path),
            "learning_path": [
                {
                    "step": i+1,
                    "topic": topic,
                    "estimated_time": "30-45 mins"
                }
                for i, topic in enumerate(ordered_path)
            ]
        }
        
        return json.dumps(path, indent=2)
    except Exception as e:
        return f"Error creating learning path: {str(e)}"

@tool
def trackLearningProgress(topic: str, status: str, score: float = None) -> str:
    """
    Track progress on a learning topic.
    
    Args:
        topic: Topic being tracked
        status: Status (started/completed/in_progress/needs_review)
        score: Optional quiz score (0-100)
        
    Returns:
        Progress update confirmation
    """
    try:
        progress = {
            "topic": topic,
            "status": status,
            "score": score,
            "timestamp": getCurrentDateTime()
        }
        
        # In a real implementation, this would save to a database
        # For now, we'll use the embedding system to store progress
        progress_text = f"Progress: {topic} - {status}"
        if score:
            progress_text += f" - Score: {score}%"
        
        # Store in a special progress collection
        from app.services.agent_tools import createEmbeddingsFromText
        createEmbeddingsFromText(
            text_content=progress_text,
            collection_name="learning_progress"
        )
        
        return f"Progress tracked: {progress_text}"
    except Exception as e:
        return f"Error tracking progress: {str(e)}"

@tool
def getLearningProgress(query: str = "all progress") -> str:
    """
    Retrieve learning progress history.
    
    Args:
        query: Query to filter progress (default: "all progress")
        
    Returns:
        Progress history
    """
    try:
        from app.services.agent_tools import retrieveFromEmbeddings
        
        progress = retrieveFromEmbeddings(
            collection_name="learning_progress",
            query=query,
            n_results=20
        )
        
        return f"Learning Progress:\n{progress}"
    except Exception as e:
        return f"Error retrieving progress: {str(e)}"

@tool
def generateStudyNotes(topic: str, key_points: List[str]) -> str:
    """
    Generate structured study notes for a topic.
    
    Args:
        topic: Topic for the notes
        key_points: Key points to include
        
    Returns:
        Formatted study notes
    """
    try:
        if isinstance(key_points, str):
            key_points = json.loads(key_points) if key_points.startswith('[') else [key_points]
        
        notes = f"# Study Notes: {topic}\n\n"
        notes += f"Date: {getCurrentDateTime()}\n\n"
        notes += "## Key Concepts\n\n"
        
        for i, point in enumerate(key_points, 1):
            notes += f"{i}. {point}\n"
        
        notes += "\n## Summary\n\n"
        notes += f"These are the essential points for understanding {topic}.\n"
        
        # Store in embeddings for future retrieval
        from app.services.agent_tools import createEmbeddingsFromText
        createEmbeddingsFromText(
            text_content=notes,
            collection_name="study_notes"
        )
        
        return notes
    except Exception as e:
        return f"Error generating study notes: {str(e)}"

@tool
def evaluateQuizAnswer(question: str, user_answer: str, correct_answer: str) -> str:
    """
    Evaluate a quiz answer and provide feedback.
    
    Args:
        question: The quiz question
        user_answer: User's answer
        correct_answer: Correct answer
        
    Returns:
        Evaluation feedback
    """
    try:
        is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
        
        feedback = {
            "question": question,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "feedback": "Correct! Well done." if is_correct else 
                       f"Not quite. The correct answer is: {correct_answer}"
        }
        
        return json.dumps(feedback, indent=2)
    except Exception as e:
        return f"Error evaluating answer: {str(e)}"

@tool
def recommendNextTopic(completed_topics: List[str], available_topics: List[str]) -> str:
    """
    Recommend the next topic to learn based on completed topics.
    
    Args:
        completed_topics: Topics already completed
        available_topics: All available topics
        
    Returns:
        Recommended next topic
    """
    try:
        if isinstance(completed_topics, str):
            completed_topics = json.loads(completed_topics) if completed_topics.startswith('[') else [completed_topics]
        if isinstance(available_topics, str):
            available_topics = json.loads(available_topics) if available_topics.startswith('[') else [available_topics]
        
        # Find topics not yet completed
        remaining = [t for t in available_topics if t not in completed_topics]
        
        if not remaining:
            return "All topics completed! Great work!"
        
        recommendation = {
            "next_topic": remaining[0],
            "remaining_count": len(remaining),
            "remaining_topics": remaining,
            "completion_percentage": int((len(completed_topics) / len(available_topics)) * 100)
        }
        
        return json.dumps(recommendation, indent=2)
    except Exception as e:
        return f"Error recommending topic: {str(e)}"


## TOOLS LIST ##
tools = [
    # Search tools
    searchNewsTool,
    searchWebTool,
    
    # RAG tools
    createEmbeddingsFromText,
    createEmbeddingsFromPDF,
    createEmbeddingsFromURL,
    retrieveFromEmbeddings,
    deleteEmbeddingsCollection,
    
    # contextual tools
    getConversationSummary,
    logConversationTurn,
    
    # Utility tools
    logConversationTurn,
    readFile,
    writeFile,
    listFiles,
    getCurrentDateTime,
    parseJSON,
    calculateExpression,
    summarizeText,
    getConversationSummary,
    
    # Enhanced learning tools
    extractTopicsFromText,
    createLearningPath,    
    trackLearningProgress,
    getLearningProgress,
    generateStudyNotes,
    evaluateQuizAnswer,
    recommendNextTopic
]

__export__= [tools]