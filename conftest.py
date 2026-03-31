"""pytest conftest – makes the project root importable as a package.

This allows test files to import from the rag_mvp package using
`from rag_mvp.retriever import Retriever` etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the PARENT of the project root so `rag_mvp` package is importable
project_root = Path(__file__).resolve().parent
parent_dir = project_root.parent  # .../Desktop, so rag_mvp becomes a package

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Also expose the project root for direct-import test files (chunker, file_loader, vllm_client)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
