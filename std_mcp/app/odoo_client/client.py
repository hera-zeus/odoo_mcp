import httpx
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from app.config import settings

logger = logging.getLogger(__name__)

class OdooClient:
    """Client Odoo qui exécute les requêtes au nom de l'utilisateur authentifié"""
    
    def __init__(self):
        self.url      = settings.ODOO_URL.rstrip('/')
        self.database = settings.ODOO_DB
        self.client   = httpx.AsyncClient(timeout=30.0)
    
    async def search_read(
        self,
        model: str,
        domain: List,
        fields: List[str],
        odoo_session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Exécute une requête en utilisant la session de l'utilisateur final"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method":  "call",
                "id":      1,
                "params": {
                    "model":  model,
                    "method": "search_read",
                    "args":   [domain],
                    "kwargs": {
                        "fields": fields,
                        "limit":  limit,
                        "offset": offset
                    }
                }
            }

            headers = {"Content-Type": "application/json"}
            if odoo_session_id:
                headers["Cookie"] = f"session_id={odoo_session_id}"

            response = await self.client.post(
                f"{self.url}/web/dataset/call_kw",
                json=payload,
                headers=headers
            )

            result = response.json()

            if "error" in result:
                logger.error(f"❌ Erreur Odoo sur {model}: {result['error']}")
                return []

            return result.get("result", [])

        except Exception as e:
            logger.error(f"❌ Erreur search_read sur {model}: {e}")
            return []

    async def get_time_series(
        self,
        table: str,
        field: str,
        start_date: str,
        end_date: str,
        period: str = 'M',
        odoo_session_id: str = None
    ) -> pd.Series:
        """Extraire une série temporelle agrégée"""
        domain = [
            ['date_order', '>=', start_date],
            ['date_order', '<=', end_date],
            ['state', 'in', ['sale', 'done']]
        ]

        fields = ['date_order', field]
        data   = await self.search_read(
            table, domain, fields,
            odoo_session_id=odoo_session_id,
            limit=10000
        )

        if not data:
            return pd.Series(dtype=float)

        df             = pd.DataFrame(data)
        df['date_order'] = pd.to_datetime(df['date_order'])
        df['period']     = df['date_order'].dt.to_period(period).dt.to_timestamp()
        aggregated       = df.groupby('period')[field].sum()

        return aggregated

    async def close(self):
        await self.client.aclose()
