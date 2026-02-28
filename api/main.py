from fastapi import FastAPI, File, UploadFile
import shutil
import sys
import os

# src folder ka path add karo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import predict

# FastAPI app create karo
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Image Classification API running"}


@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    
    file_path = "temp.jpg"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(file_path)

    return {"prediction": result}