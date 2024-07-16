import io
from datetime import timedelta

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm
from PIL import Image

from auth import authenticate_user, create_access_token, get_current_active_user
from config import Settings
from llm.model import ExtractionModel
from models import Token, User

router = APIRouter()
extractor = ExtractionModel()


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application"}


@router.post("/extract")
async def extract(
    image: UploadFile = File(...), current_user: User = Depends(get_current_active_user)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        text = extractor.run(image=img)
        return {"caption": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
