import uvicorn
import sys
import os

# Ensure the root of the project is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
