##imports##
from datetime import datetime, timedelta, timezone
from fastapi import Depends,FastAPI,APIRouter
from app.routes.auth_routes import auth_router
from app.routes.agent_routes import agent_router
from app.routes.rag_routes import rag_router
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

##routes
app=FastAPI()
app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(rag_router)

# Add CORS middleware for WebSocket support
agent_router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/{user}')
def greet(user: str, post: str | None = 'sucker'):
    return [f"hello {user} !\n sup...{post}"]

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
# uvicorn app.app:app --reload   --> this works on terminal
