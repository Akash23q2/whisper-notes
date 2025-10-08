##imports##
from datetime import datetime, timedelta, timezone
from fastapi import Depends,FastAPI,APIRouter
from app.services.auth_service import auth_router
import uvicorn
app=FastAPI()
app.include_router(auth_router)

@app.get('/{user}')
def greet(user: str, post: str | None = 'sucker'):
    return [f"hello {user} !\n sup...{post}"]

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
# uvicorn app.app:app --reload   --> this works on terminal
