## imports ##
from datetime import datetime, timedelta, timezone
from typing import Annotated
import jwt
from fastapi import Depends, FastAPI, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from sqlalchemy.orm import Session
from dotenv import load_dotenv, find_dotenv
import os

from app.services.db import get_db
from app.models.auth_data import Auth
from app.models.user_data import User
from app.schemas.auth_schema import AuthData, Token, UserData, UserInDB,SignUp

## load env ##
load_dotenv(dotenv_path=find_dotenv())

## secret constants ##
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES"))

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

auth_router = APIRouter()

## methods ##
def verify_password(plain_password, hashed_password):
    # verify plain password against hashed
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password):
    # hash a password
    return password_hash.hash(password)

def get_user_from_db(username: str, db: Session = Depends(get_db)):
    # fetch a user from database by username
    auth_record = db.query(Auth).filter(Auth.username == username).first()
    if auth_record and auth_record.user:
        return UserData(
            name=auth_record.user.name,
            age=auth_record.user.age,
            location=auth_record.user.location,
            gender=auth_record.user.gender,
            disabled=False, # Assuming 'disabled' is not a field on the User model yet.
            # other fields
        )
    return None


def authenticate_user(db: Session, username: str, password: str):
    # authenticate user using real DB
    user_record = db.query(Auth).filter(Auth.username == username).first()
    if not user_record:
        return False
    if not verify_password(password, user_record.password):
        return False
    return user_record.user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    # create JWT access token
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_db)
):
    # get current user from token
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception

    user = get_user_from_db(username, db)
    if user is None:
        raise credentials_exception
    return user

# async def get_current_active_user(
#     current_user: Annotated[UserData, Depends(get_current_user)]
# ):
#     # ensure user is active
#     if current_user.disabled:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user

## routes ##

## signup route ##
@auth_router.post("/signup", response_model=dict)
async def signup(user: Annotated[SignUp,Depends()], db: Session = Depends(get_db)):
    # check if username/email exists
    existing = db.query(Auth).filter(Auth.username==user.username).first()
    if existing:
        raise HTTPException(400, "Username already exists")
    
    hashed_password = get_password_hash(user.password.get_secret_value())
    
    # Create User and Auth objects and link them
    new_user = User(
        name=user.name,
        age=user.age,
        location=user.location,
        gender=user.gender
    )
    new_auth = Auth(
        username=user.username,
        email=user.email,
        password=hashed_password,
        user=new_user  # Link the user object here
    )
    db.add(new_auth)
    db.commit()
    
    return {"msg": "User created successfully"}



@auth_router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    # login endpoint to get JWT token
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # The user object from authenticate_user is now a User model instance
    # We need the username from the related Auth model.
    access_token = create_access_token(data={"sub": user.auth.username}, expires_delta=access_token_expires)
    return Token(access_token=access_token, token_type="bearer")

@auth_router.get("/users/me/", response_model=UserData)
async def read_users_me(
    current_user: Annotated[UserData, Depends(get_current_user)]
):
    # get current logged-in user info
    return current_user
