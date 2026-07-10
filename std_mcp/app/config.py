import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Settings:
    # Odoo
    ODOO_URL: str = os.getenv("ODOO_URL", "https://st19.sky-erp.app")
    ODOO_DB: str = os.getenv("ODOO_DB", "st19")
    #ODOO_USERNAME: str = os.getenv("ODOO_USERNAME", "api_user")
    #ODOO_PASSWORD: str = os.getenv("ODOO_PASSWORD", "")
    
    # LLM
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "anthropic/claude-sonnet-4-6")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    

    # Super Admin (authentification interne à l'application)
    SUPER_ADMIN_USERNAME: str      = os.getenv("SUPER_ADMIN_USERNAME", "admin")
    SUPER_ADMIN_PASSWORD_HASH: str = os.getenv("SUPER_ADMIN_PASSWORD_HASH", "")
    SUPER_ADMIN_EMAIL: str         = os.getenv("SUPER_ADMIN_EMAIL", "")

    # Application
    API_KEY: str = os.getenv("API_KEY", "")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENV: str = os.getenv("ENV", "development")
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CACHE_DIR: str = os.path.join(BASE_DIR, "data", "cache")
    LOGS_DIR: str = os.path.join(BASE_DIR, "data", "logs")

settings = Settings()

# Créer les dossiers nécessaires
os.makedirs(settings.CACHE_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)
