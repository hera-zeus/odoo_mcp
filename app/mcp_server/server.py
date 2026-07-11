import json
import logging
from fastmcp import FastMCP
from app.odoo_client.client import OdooClient
from app.config import settings

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="ST Digital MCP Server",
    instructions=(
        "Serveur MCP connecté à l'ERP Odoo de ST Digital. "
        "Tu peux interroger n'importe quel modèle Odoo, lister les ressources disponibles "
        "et générer des prévisions de ventes par lissage exponentiel."
    )
)

_odoo          = OdooClient()
_admin_session: str | None = None
company_info: dict = {}   # nom + devise lus depuis res.company au démarrage


async def init_admin_session() -> None:
    """Initialise la session admin Odoo et charge les infos de l'entreprise."""
    global _admin_session, company_info
    if not settings.ODOO_ADMIN_LOGIN or not settings.ODOO_ADMIN_KEY:
        logger.warning("ODOO_ADMIN_LOGIN / ODOO_ADMIN_KEY non configurés — outils MCP limités")
        return
    _admin_session = await _odoo.authenticate(settings.ODOO_ADMIN_LOGIN, settings.ODOO_ADMIN_KEY)
    if not _admin_session:
        logger.error("Impossible d'initialiser la session admin MCP")
        return
    logger.info("Session admin MCP Odoo initialisée")
    await _fetch_company_info()


async def _fetch_company_info() -> None:
    """Lit le nom et la devise de la société principale depuis res.company."""
    global company_info
    try:
        records = await _odoo.search_read(
            model="res.company",
            domain=[["active", "=", True]],
            fields=["name", "currency_id"],
            odoo_session_id=_admin_session,
            limit=1
        )
        if records:
            company = records[0]
            currency = company.get("currency_id")
            company_info = {
                "name":     company.get("name", settings.COMPANY_NAME),
                "currency": currency[1] if isinstance(currency, list) else settings.COMPANY_CURRENCY
            }
            logger.info(f"Société chargée : {company_info['name']} — devise : {company_info['currency']}")
        else:
            company_info = {"name": settings.COMPANY_NAME, "currency": settings.COMPANY_CURRENCY}
    except Exception as e:
        logger.error(f"Erreur lecture res.company: {e}")
        company_info = {"name": settings.COMPANY_NAME, "currency": settings.COMPANY_CURRENCY}


# ──────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────

@mcp.tool()
async def search_odoo_records(
    model: str,
    fields: list[str],
    domain: list = [],
    limit: int = 20
) -> str:
    """
    Recherche et retourne des enregistrements depuis n'importe quel modèle Odoo.
    Utilise cet outil pour répondre à des questions sur les ventes, achats,
    employés, factures, stocks, contacts, budgets, etc.

    Args:
        model:  Nom technique du modèle Odoo (ex: sale.order, account.move, hr.employee)
        fields: Champs à retourner (ex: ['name', 'amount_total', 'state'])
        domain: Filtres Odoo (ex: [['state','=','sale'], ['date_order','>=','2026-01-01']])
        limit:  Nombre maximum d'enregistrements (défaut: 20)
    """
    records = await _odoo.search_read(
        model=model, domain=domain, fields=fields,
        odoo_session_id=_admin_session, limit=limit
    )
    return json.dumps(
        {"model": model, "count": len(records), "records": records},
        ensure_ascii=False, default=str
    )


@mcp.tool()
async def get_odoo_models(keyword: str = "") -> str:
    """
    Retourne la liste des modèles Odoo disponibles.
    Utilise cet outil quand tu ne sais pas quel modèle interroger
    pour répondre à une question métier.

    Args:
        keyword: Mot-clé pour filtrer (ex: 'sale', 'hr', 'account')
    """
    records = await _odoo.search_read(
        model="ir.model",
        domain=[["model", "ilike", keyword]] if keyword else [],
        fields=["model", "name"],
        odoo_session_id=_admin_session,
        limit=50
    )
    return json.dumps({"count": len(records), "models": records}, ensure_ascii=False)


@mcp.tool()
async def get_forecast(
    model: str,
    field: str,
    date_field: str,
    domain: list[list] = [],
    periods: int = 6,
    period: str = "M",
    start_date: str = "",
    end_date: str = ""
) -> str:
    """
    Génère des prévisions via lissage exponentiel Holt-Winters (ETS) sur n'importe
    quel champ numérique d'un modèle Odoo.
    Utilise cet outil quand l'utilisateur demande des prévisions, projections
    ou tendances futures (ventes, achats, charges, consommation stock, etc.).

    Args:
        model:      Modèle Odoo source (ex: sale.order, account.move, purchase.order, hr.payslip)
        field:      Champ numérique à prévoir (ex: amount_total, net_wage, product_uom_qty)
        date_field: Champ date du modèle (ex: date_order, invoice_date, date_from)
        start_date: Début de l'historique (YYYY-MM-DD)
        end_date:   Fin de l'historique (YYYY-MM-DD)
        domain:     Filtres additionnels (ex: [['state','=','posted']])
        periods:    Nombre de périodes à prévoir (défaut: 6)
        period:     Granularité 'D', 'W', 'M', 'Q' (défaut: 'M')
    """
    from app.forecasting.engine import ForecastEngine
    from datetime import datetime as _dt
    engine = ForecastEngine()

    today      = _dt.now()
    _end_date  = end_date   or today.strftime("%Y-%m-%d")
    _start_date = start_date
    if not _start_date:
        first_rec = await _odoo.search_read(
            model=model,
            domain=[[date_field, "!=", False]] + list(domain),
            fields=[date_field],
            odoo_session_id=_admin_session,
            limit=1,
            order=f"{date_field} asc"
        )
        if first_rec and first_rec[0].get(date_field):
            _start_date = str(first_rec[0][date_field])[:10]
        else:
            _start_date = "2020-01-01"
        logger.info(f"get_forecast: start_date auto-détecté = {_start_date}")

    series = await _odoo.get_time_series(
        table=model, field=field, date_field=date_field,
        domain=domain, start_date=_start_date, end_date=_end_date,
        period=period, odoo_session_id=_admin_session
    )

    if series.empty:
        return json.dumps({"error": f"Aucune donnée trouvée pour {model}.{field}."})

    result  = engine.forecast_prophet(series, periods=periods)
    metrics = engine.calculate_metrics(result["series_clean"], result["fitted"])

    model_info = result.get("model_info", {})

    # Tableau prévision : valeur centrale + bornes de confiance Prophet
    if "forecast_lower" in result and "forecast_upper" in result:
        forecast_dict = {
            str(d.date()): {
                "valeur":      round(float(v), 2),
                "borne_basse": round(float(lo), 2),
                "borne_haute": round(float(hi), 2),
            }
            for d, v, lo, hi in zip(
                result["forecast_dates"], result["forecast"],
                result["forecast_lower"], result["forecast_upper"]
            )
        }
    else:
        forecast_dict = {
            str(d.date()): round(float(v), 2)
            for d, v in zip(result["forecast_dates"], result["forecast"])
        }

    response = {
        "type":        "forecast",
        "odoo_model":  model,
        "odoo_field":  field,
        "algo":        model_info.get("algo", "Prophet"),
        "n_history":   model_info.get("n_points"),
        "periods":     periods,
        "granularity": period,
        "forecast":    forecast_dict,
        "metrics": {
            "mae":      round(metrics.get("mae", 0), 2),
            "smape":    round(metrics.get("smape", 0), 2),
            "mape":     metrics.get("mape"),
            "n_points": metrics.get("n_points", 0),
        }
    }

    return json.dumps(response, ensure_ascii=False, default=str)


def create_mcp_server() -> FastMCP:
    """Retourne l'instance FastMCP configurée (tools + resources enregistrés au niveau module)."""
    return mcp
