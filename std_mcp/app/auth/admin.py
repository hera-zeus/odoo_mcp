import bcrypt
import uuid
import logging
from typing import Optional, Dict
from app.config import settings

logger = logging.getLogger(__name__)

# Sessions admin actives (en mémoire — même logique que les sessions utilisateur)
admin_sessions: Dict[str, dict] = {}


def verify_admin_credentials(username: str, password: str) -> bool:
    """Vérifie les identifiants du super admin contre le hash bcrypt"""
    if username != settings.SUPER_ADMIN_USERNAME:
        return False

    try:
        return bcrypt.checkpw(
            password.encode('utf-8'),
            settings.SUPER_ADMIN_PASSWORD_HASH.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"❌ Erreur vérification mot de passe admin: {e}")
        return False


def create_admin_session(username: str) -> str:
    """Crée une session admin et retourne son ID"""
    admin_session_id = str(uuid.uuid4())
    admin_sessions[admin_session_id] = {
        "username": username,
        "email":    settings.SUPER_ADMIN_EMAIL,
        "role":     "super_admin"
    }
    logger.info(f"✅ Session admin créée pour {username}")
    return admin_session_id


def get_admin_session(admin_session_id: str) -> Optional[dict]:
    """Récupère les infos d'une session admin"""
    return admin_sessions.get(admin_session_id)


def revoke_admin_session(admin_session_id: str) -> bool:
    """Révoque une session admin"""
    if admin_session_id in admin_sessions:
        del admin_sessions[admin_session_id]
        return True
    return False
