from __future__ import annotations

from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_papers(data_dir: Path | str = Path("data/papers")) -> List[Document]:
    """Load PDF papers into a LlamaIndex document list for RAG workflows."""

    data_path = Path(data_dir)
    reader = SimpleDirectoryReader(input_dir=str(data_path), recursive=True)
    return reader.load_data()
