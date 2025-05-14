from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
from isegm.utils import exp
from isegm.inference import utils
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import io
import base64
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
import logging
import traceback
import json
import sys

# Configure logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to store session data
model = None
current_image = None
current_mask = None
clicks_list = []
device = None
predictor = None
clicker_obj = clicker.Clicker()  # Initialize clicker at global level
object_count = 0  # Track number of finished objects

class SessionConfig(BaseModel):
    checkpoint: str = "models/coco_lvis_h18_baseline.pth"  # Default value
    gpu: Optional[int] = 0
    cpu: Optional[bool] = True  # Default to CPU
    limit_longest_size: Optional[int] = 800
    cfg: Optional[str] = "config.yml"

class ClickData(BaseModel):
    x: int
    y: int
    is_positive: bool

async def initialize_default_session():
    """Initialize session with default parameters when server starts"""
    global model, device, predictor, object_count
    
    try:
        print("Initializing default session")
        logger.info("Initializing default session")
        
        # Default configuration
        config = SessionConfig()  # Use default values
        
        # Load configuration
        cfg = exp.load_config_file(config.cfg, return_edict=True)
        print("Configuration loaded successfully")
        logger.info("Configuration loaded successfully")
        
        # Set device
        device = torch.device('cpu')
        print(f"Using device: {device}")
        logger.info(f"Using device: {device}")
            
        # Load model
        checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, config.checkpoint)
        print(f"Loading model from checkpoint: {checkpoint_path}")
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True)
        print("Model loaded successfully")
        logger.info("Model loaded successfully")
        
        # Initialize predictor
        print("Initializing predictor")
        logger.info("Initializing predictor")
        predictor = get_predictor(model, brs_mode='NoBRS', device=device)
        
        # Reset clicker and object count
        clicker_obj.reset_clicks()
        object_count = 0
        
        print("Default session initialized successfully")
        logger.info("Default session initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing default session: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"Error initializing default session: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize default session when server starts"""
    success = await initialize_default_session()
    if not success:
        print("Failed to initialize default session")
        logger.error("Failed to initialize default session")

@app.post("/init-session")
async def init_session(config: SessionConfig):
    global model, device, predictor, object_count
    
    try:
        logger.info(f"Initializing session with config: {config}")
        
        # Load configuration
        cfg = exp.load_config_file(config.cfg, return_edict=True)
        logger.info("Configuration loaded successfully")
        
        # Set device
        if config.cpu:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{config.gpu}')
        logger.info(f"Using device: {device}")
            
        # Load model
        checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, config.checkpoint)
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True)
        logger.info("Model loaded successfully")
        
        # Initialize predictor
        logger.info("Initializing predictor")
        predictor = get_predictor(model, brs_mode='NoBRS', device=device)
        
        # Reset clicker and object count
        clicker_obj.reset_clicks()
        object_count = 0
        
        logger.info("Session initialized successfully")
        return {
            "status": "success",
            "message": "Session initialized successfully",
            "device": str(device),
            "checkpoint": config.checkpoint
        }
        
    except Exception as e:
        logger.error(f"Error initializing session: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    global current_image, current_mask, clicks_list, predictor, object_count
    
    try:
        logger.info("Processing image upload")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        logger.info(f"Image decoded successfully. Shape: {image.shape}")
            
        # Store the image
        current_image = image
        current_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        clicks_list = []
        clicker_obj.reset_clicks()
        object_count = 0
        
        # Set image in predictor
        logger.info("Setting image in predictor")
        predictor.set_input_image(image)
        
        logger.info("Image uploaded and processed successfully")
        return {
            "status": "success",
            "message": "Image uploaded successfully",
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/click")
async def process_click(request: Request):
    global current_image, current_mask, clicks_list, predictor
    
    try:
        # Log raw request body
        body = await request.body()
        print("Raw request body:", body.decode())
        logger.info(f"Raw request body: {body.decode()}")
        
        # Parse JSON body manually
        click_data = json.loads(body.decode())
        print("Parsed click data:", click_data)
        logger.info(f"Parsed click data: {click_data}")
        
        # Validate required fields
        if 'x' not in click_data or 'y' not in click_data or 'is_positive' not in click_data:
            raise HTTPException(status_code=422, detail="Missing required fields: x, y, is_positive")
        
        # Convert to ClickData model
        click = ClickData(
            x=int(click_data['x']),
            y=int(click_data['y']),
            is_positive=bool(click_data['is_positive'])
        )
        
        print("Converted click data:", click.dict())
        logger.info(f"Converted click data: {click.dict()}")
        
        if current_image is None:
            print("Error: No image uploaded")
            logger.error("No image uploaded")
            raise HTTPException(status_code=400, detail="No image uploaded")
        if model is None:
            print("Error: Model not initialized")
            logger.error("Model not initialized")
            raise HTTPException(status_code=400, detail="Model not initialized")
        if predictor is None:
            print("Error: Predictor not initialized")
            logger.error("Predictor not initialized")
            raise HTTPException(status_code=400, detail="Predictor not initialized")
            
        # Add click - Note: coords should be (y, x) for the model
        click_obj = clicker.Click(is_positive=click.is_positive, coords=(click.y, click.x))
        clicker_obj.add_click(click_obj)
        
        # Get prediction
        print("Getting prediction from model")
        logger.info("Getting prediction from model")
        pred_mask = predictor.get_prediction(clicker_obj)
        
        # Update current mask
        current_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        print(f"Click processed successfully. Total clicks: {len(clicker_obj.clicks_list)}")
        logger.info(f"Click processed successfully. Total clicks: {len(clicker_obj.clicks_list)}")
        return {
            "status": "success",
            "message": "Click processed successfully",
            "clicks_count": len(clicker_obj.clicks_list)
        }
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}")
        logger.error(f"Error decoding JSON: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid JSON format")
    except Exception as e:
        print(f"Error processing click: {str(e)}")
        print(traceback.format_exc())
        logger.error(f"Error processing click: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-clicks")
async def reset_clicks():
    global current_mask, predictor
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")
    if predictor is None:
        raise HTTPException(status_code=400, detail="Predictor not initialized")
        
    try:
        # Reset clicks
        clicker_obj.reset_clicks()
        
        # Reset mask
        current_mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
        
        return {
            "status": "success",
            "message": "Clicks reset successfully",
            "clicks_count": 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/undo-click")
async def undo_click():
    global current_mask, predictor
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")
    if predictor is None:
        raise HTTPException(status_code=400, detail="Predictor not initialized")
    if len(clicker_obj.clicks_list) == 0:
        raise HTTPException(status_code=400, detail="No clicks to undo")
        
    try:
        # Remove last click
        clicker_obj.undo_last_click()
        
        # Get new prediction
        if len(clicker_obj.clicks_list) > 0:
            pred_mask = predictor.get_prediction(clicker_obj)
            current_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        else:
            current_mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
        
        return {
            "status": "success",
            "message": "Click undone successfully",
            "clicks_count": len(clicker_obj.clicks_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finish-object")
async def finish_object():
    global current_mask, object_count
    
    if current_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")
    if len(clicker_obj.clicks_list) == 0:
        raise HTTPException(status_code=400, detail="No object to finish")
        
    try:
        # Increment object count
        object_count += 1
        
        # Reset clicks for next object
        clicker_obj.reset_clicks()
        
        return {
            "status": "success",
            "message": "Object finished successfully",
            "object_count": object_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-mask")
async def get_mask():
    global current_image, current_mask
    
    try:
        logger.info("Getting mask")
        
        if current_image is None:
            logger.error("No image uploaded")
            raise HTTPException(status_code=400, detail="No image uploaded")
        if current_mask is None:
            logger.error("No mask generated")
            raise HTTPException(status_code=400, detail="No mask generated")
            
        # Create overlay image
        overlay = current_image.copy()
        overlay[current_mask > 0] = overlay[current_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', overlay)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("Mask generated successfully")
        return {
            "status": "success",
            "image": f"data:image/png;base64,{base64_image}"
        }
        
    except Exception as e:
        logger.error(f"Error getting mask: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-pure-mask")
async def get_pure_mask():
    global current_image, current_mask
    
    try:
        logger.info("Getting pure mask")
        
        if current_image is None:
            logger.error("No image uploaded")
            raise HTTPException(status_code=400, detail="No image uploaded")
        if current_mask is None:
            logger.error("No mask generated")
            raise HTTPException(status_code=400, detail="No mask generated")
            
        # Convert mask to RGB (white for foreground, black for background)
        mask_rgb = np.zeros((current_mask.shape[0], current_mask.shape[1], 3), dtype=np.uint8)
        mask_rgb[current_mask > 0] = [255, 255, 255]  # White for foreground
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', mask_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("Pure mask generated successfully")
        return {
            "status": "success",
            "image": f"data:image/png;base64,{base64_image}"
        }
        
    except Exception as e:
        logger.error(f"Error getting pure mask: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-segmented-image")
async def get_segmented_image():
    global current_image, current_mask
    
    try:
        logger.info("Getting segmented image")
        
        if current_image is None:
            logger.error("No image uploaded")
            raise HTTPException(status_code=400, detail="No image uploaded")
        if current_mask is None:
            logger.error("No mask generated")
            raise HTTPException(status_code=400, detail="No mask generated")
            
        # Create RGBA image (with alpha channel)
        segmented_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on mask
        segmented_image[:, :, 3] = current_mask  # Set alpha channel (0 for transparent, 255 for opaque)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', segmented_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("Segmented image generated successfully")
        return {
            "status": "success",
            "image": f"data:image/png;base64,{base64_image}"
        }
        
    except Exception as e:
        logger.error(f"Error getting segmented image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
