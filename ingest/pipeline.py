from __future__ import annotations
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Settings
from db.vector_store import VectorStore


# File types this pipeline 
SUPPORTED_EXTENSIONS={'.pdf', 'docx', '.txt', '.md'}

def load_file(file_path: str) -> str:
    "load a single file and return a list of langchain docs"
    
    path  = Path(file_path)