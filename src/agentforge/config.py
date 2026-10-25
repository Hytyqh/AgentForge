import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"
DEFAULT_EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBED_MODEL = "text-embedding-v4"


class Settings:
    api_key: str
    base_url: str
    chat_model: str
    embed_api_key: str
    embed_base_url: str
    embed_model: str
    workspace_root: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.environ.get("AGENTFORGE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "缺少 AGENTFORGE_API_KEY：复制 .env.example 为 .env 并填入百炼 API Key"
            )
        s = cls()
        s.api_key = api_key
        s.base_url = os.environ.get("AGENTFORGE_BASE_URL", DEFAULT_BASE_URL)
        s.chat_model = os.environ.get("AGENTFORGE_CHAT_MODEL", DEFAULT_MODEL)
        s.embed_api_key = os.environ.get("AGENTFORGE_EMBED_API_KEY", "")
        s.embed_base_url = os.environ.get("AGENTFORGE_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL)
        s.embed_model = os.environ.get("AGENTFORGE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        s.workspace_root = os.environ.get("AGENTFORGE_WORKSPACE", os.getcwd())
        return s
