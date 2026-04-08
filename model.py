import numpy as np
import cv2
import base64
from ultralytics import YOLO


def load_model(model_path: str):
    """Load YOLOv8 model from path."""
    model = YOLO(model_path)
    return model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to OpenCV image array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG or PNG.")

    return img


def predict_board(model, image_bytes: bytes) -> dict:
    """Run inference on image bytes and return annotated result."""
    img = preprocess_image(image_bytes)
    
    # Run inference
    results = model(img, conf=0.1)

    if not results or len(results) == 0:
        raise ValueError("Model returned no results.")
    
    result = results[0]
    
    # OBB model uses .obb instead of .boxes
    if result.obb is None or len(result.obb) == 0:
        raise ValueError("No detections found in image.")

    # Get detections from OBB
    detections = result.obb.data.tolist()

    # Plot still works the same
    drawn_image = result.plot()

    success, buffer = cv2.imencode(".png", drawn_image)
    if not success:
        raise RuntimeError("Failed to encode result image.")

    return {
        "image_bytes": buffer.tobytes(),
        "image_base64": base64.b64encode(buffer).decode("utf-8"),
        "detections": detections,
    }