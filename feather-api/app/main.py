from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import feather_db
from feather_db import DB, Metadata, ContextType, ScoringConfig
import os
import json
import shutil
import time

app = FastAPI(title="Feather DB Cloud API", version="0.4.0")

# --- Configuration ---
DB_PATH = os.getenv("FEATHER_DB_PATH", "cloud_v1.feather")
DB_DIM = int(os.getenv("FEATHER_DB_DIM", "768"))

# Global DB Instance
db_instance = None

@app.on_event("startup")
def startup_event():
    global db_instance
    print(f"Loading Feather DB from {DB_PATH}...")
    db_instance = DB.open(DB_PATH, dim=DB_DIM)

@app.get("/")
def health_check():
    return {"status": "online", "version": feather_db.__version__}
