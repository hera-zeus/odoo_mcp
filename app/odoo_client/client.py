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
        date_field: str = 'date_order',
        domain: list = None,
        odoo_session_id: str = None
    ) -> pd.Series:
        """
        Extraire une série temporelle agrégée depuis n'importe quel modèle Odoo.

        Args:
            table:      Modèle Odoo (ex: sale.order, account.move, hr.payslip)
            field:      Champ numérique à agréger (ex: amount_total, net_wage)
            start_date: Début de la période (YYYY-MM-DD)
            end_date:   Fin de la période (YYYY-MM-DD)
            period:     Granularité : 'D', 'W', 'M', 'Q'
            date_field: Champ date du modèle (ex: date_order, invoice_date, date_from)
            domain:     Filtres additionnels Odoo (ex: [['state','=','posted']])
        """
        base_domain = [
            [date_field, '>=', start_date],
            [date_field, '<=', end_date],
        ]
        if domain:
            base_domain.extend(domain)

        fields = [date_field, field]

        # Pagination pour ne pas dépasser les limites Odoo sur de gros volumes
        all_data = []
        offset   = 0
        batch    = 5000
        while True:
            chunk = await self.search_read(
                table, base_domain, fields,
                odoo_session_id=odoo_session_id,
                limit=batch,
                offset=offset
            )
            if not chunk:
                break
            all_data.extend(chunk)
            if len(chunk) < batch:
                break
            offset += batch

        logger.info(f"{len(all_data)} enregistrements récupérés pour {table}.{field}")

        if not all_data:
            return pd.Series(dtype=float)

        df                = pd.DataFrame(all_data)
        df[date_field]    = pd.to_datetime(df[date_field])
        df['period']      = df[date_field].dt.to_period(period).dt.to_timestamp()
        aggregated        = df.groupby('period')[field].sum()

        # Index complet et régulier — les trous brisent pd.infer_freq et dégradent l'ETS
        freq_map   = {'D': 'D', 'W': 'W-MON', 'M': 'MS', 'Q': 'QS'}
        freq       = freq_map.get(period, 'MS')
        full_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        aggregated = aggregated.reindex(full_index, fill_value=0)

        n_zeros = (aggregated == 0).sum()
        if n_zeros > 0:
            logger.warning(f"{n_zeros}/{len(aggregated)} périodes à zéro dans {table}.{field}")

        return aggregated

    async def authenticate(self, login: str, password: str) -> str | None:
        """Authentifie un compte Odoo et retourne son session_id"""
        try:
            response = await self.client.post(
                f"{self.url}/web/session/authenticate",
                json={
                    "jsonrpc": "2.0", "method": "call", "id": 1,
                    "params": {"db": self.database, "login": login, "password": password}
                }
            )
            result = response.json()
            if not result.get("result", {}).get("uid"):
                logger.warning(f"❌ Échec authentification Odoo pour {login}")
                return None

            session_id = response.cookies.get("session_id")
            if not session_id:
                for part in response.headers.get("set-cookie", "").split(";"):
                    if part.strip().startswith("session_id="):
                        session_id = part.strip().split("=", 1)[1]
                        break

            logger.info(f"✅ Authentification Odoo OK pour {login}")
            return session_id
        except Exception as e:
            logger.error(f"❌ Erreur authenticate({login}): {e}")
            return None

    async def close(self):
        await self.client.aclose()
