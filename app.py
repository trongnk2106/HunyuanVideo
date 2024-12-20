from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from loguru import logger
from datetime import datetime
import os
import time

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.config import parse_args
from load_model import LoadModel


app = FastAPI()

model = LoadModel()

# Define input schema with docstrings for parameters
class GenerateVideoPayload(BaseModel):
    prompt: str = Field(..., description="Text prompt to guide video generation.")
    video_size: list[int] = Field(..., description="Size of the generated video [height, width].")
    video_length: int = Field(..., description="Length of the generated video in frames.")
    seed: int = Field(..., description="Random seed for reproducibility.")
    infer_steps: int = Field(..., description="Number of inference steps for the model.")
    cfg_scale: float = Field(..., description="Guidance scale for controlling generation strength.")
    num_videos: int = Field(..., description="Number of videos to generate.")
    flow_shift: float = Field(..., description="Shift for controlling temporal consistency of frames.")
    embedded_cfg_scale: float = Field(..., description="Embedded guidance scale for additional fine-tuning.")

@app.get("/")
def read_root():
    """
    Root route for health check.
    """
    return {"message": "Hello, this is the video generation API!"}


@app.post("/generate-video")
def generate_video(payload: GenerateVideoPayload):
    """
    Generate videos based on the input parameters provided in the payload.

    Args:
        payload (GenerateVideoPayload): The parameters for video generation.

    Returns:
        JSON response containing the paths to the generated videos.
    """
    # Parse args from payload
    logger.info(f"Received payload: {payload}")
    
    # Check if model path exists
   

    # Generate videos
    try:
        outputs = model.hunyuan_video_sampler.predict(
            prompt=payload.prompt,
            height=payload.video_size[0],
            width=payload.video_size[1],
            video_length=payload.video_length,
            seed=payload.seed,
            negative_prompt=model.neg_prompt,
            infer_steps=payload.infer_steps,
            guidance_scale=model.cfg_scale,
            num_videos_per_prompt=payload.num_videos,
            flow_shift=payload.flow_shift,
            batch_size=model.batch_size,
            embedded_guidance_scale=payload.embedded_cfg_scale
        )
    except Exception as e:
        logger.error(f"Error during video generation: {e}")
        raise HTTPException(status_code=500, detail="Error during video generation")

    # Save videos
    res = model.save_output(
        outputs = outputs
    )
    
    if isinstance(res, dict):
        return res['Output_Path']
    else:
        return res 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
