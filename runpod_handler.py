# runpod_handler.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uuid
import subprocess
import shutil
from pathlib import Path
import tempfile
import glob
from datetime import datetime

app = FastAPI(title="Ditto TalkingHead API", version="1.0")

# --- Mount static files ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Ditto TalkingHead API"}

# --- Root endpoint to serve the UI ---
@app.get("/")
async def root():
    """Serve the upload UI."""
    return FileResponse("static/index.html")

# --- File listing endpoint ---
@app.get("/files")
async def list_files():
    """List all generated video files."""
    try:
        if not os.path.exists(OUTPUT_DIR):
            return {"files": []}
        
        files = []
        for file_path in glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")):
            file_stat = os.stat(file_path)
            files.append({
                "filename": os.path.basename(file_path),
                "size": file_stat.st_size,
                "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "download_url": f"/download/{os.path.basename(file_path)}"
            })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)
        return {"files": files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

# --- File download endpoint ---
@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a specific video file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename
    )

# --- Environment and paths ---
CHECKPOINT_DIR = "/workspace/checkpoints/ditto_trt_Ampere_Plus"
CONFIG_PKL = "/workspace/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
OUTPUT_DIR = "/workspace/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- File validation ---
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension."""
    return Path(filename).suffix.lower() in allowed_extensions

# --- Core inference function ---
def run_inference(source_img, audio_file):
    """Run Ditto inference with uploaded files."""
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
    cmd = [
        "python3", "inference.py",
        "--data_root", CHECKPOINT_DIR,
        "--cfg_pkl", CONFIG_PKL,
        "--audio_path", audio_file,
        "--source_path", source_img,
        "--output_path", output_path,
    ]
    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
# --- FastAPI Endpoint ---
@app.post("/inference")
async def inference(
    source_image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """Upload source image and audio file to generate talking head video."""
    
    # Validate file types
    if not validate_file_type(source_image.filename, ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )
    
    if not validate_file_type(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{source_image.filename}")
    audio_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{audio.filename}")
    
    try:
        # Save uploaded files
        with open(img_path, "wb") as f:
            f.write(await source_image.read())
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        # Run inference
        result_path = run_inference(img_path, audio_path)
        
        # Return the generated video
        return FileResponse(
            result_path, 
            media_type="video/mp4",
            filename=f"talking_head_{uuid.uuid4().hex}.mp4"
        )
    
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- RunPod handler (for async jobs) ---
def handler(event):
    """
    RunPod handler format.
    Expects event.input to contain:
      {
        "source_path": "path/to/image.png",
        "audio_path": "path/to/audio.wav"
      }
    """
    try:
        source = event["input"].get("source_path")
        audio = event["input"].get("audio_path")
        
        if not source or not audio:
            return {"error": "Missing source_path or audio_path in input"}
        
        result = run_inference(source, audio)
        return {"output_path": result}
    
    except Exception as e:
        return {"error": str(e)}

# Start the appropriate server based on environment
# name == "__main__":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)