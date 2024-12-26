import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent
sys.path.extend([
    str(root_dir),
    str(root_dir / "apps"),
    str(root_dir / "shared")
])

from fastapi import FastAPI
from apps.knowledge_base.app.api.router import kb_router

app = FastAPI(title="Knowledge Base API")

# Mount only KB app
app.include_router(kb_router, prefix="/kb")

if __name__ == "__main__":
    import uvicorn

    # Simplified to run single server
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True) 