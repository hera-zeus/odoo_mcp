import httpx
import logging
from typing import Optional, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

class OdooUserAuth:
    """Gestionnaire d'authentification déléguée à Odoo"""
    
    def __init__(self):
        self.odoo_url = settings.ODOO_URL.rstrip('/')
        self.database = settings.ODOO_DB
        self.client   = httpx.AsyncClient(timeout=30.0)
    
    async def authenticate_user(self, login: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Authentifie l'utilisateur directement auprès d'Odoo via session cookie"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method":  "call",
                "id":      1,
                "params": {
                    "db":       self.database,
                    "login":    login,
                    "password": api_key
                }
            }

            response = await self.client.post(
                f"{self.odoo_url}/web/session/authenticate",
                json=payload
            )

            result = response.json()
            logger.info(f"📥 Réponse Odoo auth: uid={result.get('result', {}).get('uid')}")

            if result.get('result', {}).get('uid'):
                user_data = result['result']

                # Récupérer le session_id depuis les cookies
                session_id = response.cookies.get('session_id')

                # Fallback : chercher dans Set-Cookie header
                if not session_id:
                    set_cookie = response.headers.get('set-cookie', '')
                    for part in set_cookie.split(';'):
                        part = part.strip()
                        if part.startswith('session_id='):
                            session_id = part.split('=', 1)[1]
                            break

                logger.info(f"✅ Auth OK: {user_data.get('name')} | session_id={'OK' if session_id else 'MANQUANT'}")

                return {
                    "uid":              user_data['uid'],
                    "name":             user_data.get('name', 'Unknown'),
                    "email":            login,
                    "odoo_session_id":  session_id,
                    "context":          user_data.get('user_context', {})
                }
            else:
                logger.warning("❌ Échec auth Odoo — login ou mot de passe invalide")
                return None

        except Exception as e:
            logger.error(f"❌ Erreur authentification: {e}")
            return None

    async def close(self):
        await self.client.aclose()
