import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the .env next to the backend folder
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INDEX_DIR = os.getenv("INDEX_DIR", str((Path(__file__).resolve().parents[1] / "storage" / "index")))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
