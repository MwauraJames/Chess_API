from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import os
from io import BytesIO
import cv2

from model import load_model, predict_board

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Load model once at startup
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    model = None


@app.get('/')
def health_check():
    return {"message": "Chess board analyzer is running!"}


@app.post(
    "/predict/image",
    summary="Upload chessboard → annotated image with all pieces highlighted",
    responses={
        200: {"content": {"image/png": {}}, "description": "Annotated PNG with pieces highlighted"},
        400: {"description": "Bad request (wrong file type, too large, invalid position)"},
        500: {"description": "Server / model error"},
    },
)
async def predict_image(file: UploadFile = File(..., description="Chessboard Picture (JPG or PNG)")):
    # 1. Validate type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only JPG and PNG are accepted."
        )

    # 2. Read bytes
    image_bytes = await file.read()

    # 3. Validate size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum is 10 MB.")

    # 4. Check model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load at startup.")

    try:
        result = predict_board(model, image_bytes)  # ← Pass the MODEL object, not path

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # 5. Return the annotated image as PNG
    return StreamingResponse(
        BytesIO(result["image_bytes"]),
        media_type="image/png"
    )