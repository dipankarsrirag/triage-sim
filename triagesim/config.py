from dotenv import load_dotenv
import os

# load from the .env file (project root by default)
load_dotenv()  # will load .env into os.environ

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ENABLE_LLM_DETECTORS = False
