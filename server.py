import os
from typing import Dict
import modal
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import wandb
import numpy as np
from PIL import Image
import io
from tensorflow import keras
from tensorflow.keras import backend as K


# Update image to include wandb package
image = (modal.Image.debian_slim()
         .pip_install("fastapi", "wandb", "torch", "tensorflow", "h5py", "pillow", "numpy", "python-multipart"))
vol = modal.Volume.from_name("my-volume", create_if_missing=True)
# Create named app
app = modal.App("wandb-webhook")
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_image(file: UploadFile) -> np.ndarray:
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Resize to 128x128
    img = img.resize((128, 128))
    
    # Convert to numpy array and preprocess
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

@web_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Process the uploaded image
        img_array = await process_image(file)
        
        # Load model
        download_dir = "/data/artifacts"
        model_path = os.path.join(download_dir, "model.h5")
        
        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": "Model not found. Please upload model first."
            }
            
        model = keras.models.load_model(model_path)
        
        # Get prediction
        predictions = model.predict(img_array)
        
        return {
            "status": "success",
            "predictions": predictions.tolist()  # Convert numpy array to list
        }
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

@web_app.post("/")
async def webhook(request: Request):
    payload = await request.json()
    
    # Extract W&B specific information
    client_payload = payload.get("client_payload", {})
    entity_name = client_payload.get("entity_name")
    project_name = client_payload.get("project_name")
    artifact_name = client_payload.get("artifact_collection_name")
    artifact_version = client_payload.get("artifact_version_string")
    
    try:
        # Initialize W&B
        api = wandb.Api()
        
        # Construct artifact identifier with correct format
        artifact_path = f"{artifact_version}"
        print(f"Attempting to download artifact: {artifact_path}")
        
        # Download the artifact
        artifact = api.artifact(artifact_path)
        download_dir = "/data/artifacts"
        os.makedirs(download_dir, exist_ok=True)
        artifact_dir = artifact.download(root=download_dir)

        # Initialize parameters
        total_params = 0
        trainable_params = 0
        
        # Load the model and count parameters
        model_path = os.path.join(download_dir, "model.h5") 
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                total_params = model.count_params()
                trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
                print(f"Model loaded successfully. Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
            except Exception as model_error:
                print(f"Error loading model: {str(model_error)}")
                return {
                    "status": "error",
                    "message": f"Error loading model: {str(model_error)}"
                }
        else:
            print(f"Model file not found at: {model_path}")
            return {
                "status": "error",
                "message": f"Model file not found at: {model_path}"
            }
        
        print(f"Successfully downloaded artifact to {download_dir}")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        return {
            "status": "success", 
            "message": f"Artifact downloaded to {download_dir}",
            "model_stats": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            }
        }
        
    except Exception as e:
        error_msg = f"Error downloading artifact: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

@app.function(
    secrets=[
        modal.Secret.from_name("my-random-secret"),
        modal.Secret.from_name("wandb-secret") 
    ],
    image=image,
    volumes={"/data": vol}
)
@modal.asgi_app()
def serve():
    return web_app