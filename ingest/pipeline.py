from __future__ import annotations
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Settings
from db.vector_store import VectorStore


# File types this pipeline supports