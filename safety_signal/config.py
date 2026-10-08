"""Settings. The API key comes from the environment first, then a .env file. It is never printed."""
import os
from pathlib import Path
from typing import Mapping, Optional


def load_api_key(env: Optional[Mapping[str, str]] = None, dotenv_path: str = ".env") -> Optional[str]:
    env = os.environ if env is None else env
    key = env.get("OPENFDA_API_KEY")
    if key:
        return key
    try:
        lines = Path(dotenv_path).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("OPENFDA_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None
