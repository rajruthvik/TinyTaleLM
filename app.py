import os
import sys

# Append root directory to sys.path to ensure module discovery
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.serving.server import app

if __name__ == "__main__":
    import uvicorn
    print("Starting miniLLM application server at http://127.0.0.1:8000...")
    # Serve FastAPI app on port 8000 with reload enabled for development
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
