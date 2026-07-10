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


async def init_admin_session() -> None:
    """Initialise la session admin Odoo au démarrage du serveur."""
    global _admin_session
    if not settings.ODOO_ADMIN_LOGIN or not settings.ODOO_ADMIN_KEY:
        logger.warning("⚠️ ODOO_ADMIN_LOGIN / ODOO_ADMIN_KEY non configurés — outils MCP limités")
        return
    _admin_session = await _odoo.authenticate(settings.ODOO_ADMIN_LOGIN, settings.ODOO_ADMIN_KEY)
    if _admin_session:
        logger.info("✅ Session admin MCP Odoo initialisée")
    else:
        logger.error("❌ Impossible d'initialiser la session admin MCP")


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
    start_date: str,
    end_date: str,
    domain: list[list] = [],
    periods: int = 6,
    period: str = "M"
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
    engine = ForecastEngine()

    series = await _odoo.get_time_series(
        table=model, field=field, date_field=date_field,
        domain=domain, start_date=start_date, end_date=end_date,
        period=period, odoo_session_id=_admin_session
    )

    if series.empty:
        return json.dumps({"error": f"Aucune donnée trouvée pour {model}.{field}."})

    result  = engine.forecast_ets(series, periods=periods)
    metrics = engine.calculate_metrics(result["series_clean"], result["fitted"])

    forecast_dict = {
        str(d.date()): round(float(v), 2)
        for d, v in zip(result["forecast_dates"], result["forecast"])
    }

    return json.dumps({
        "type":        "forecast",
        "odoo_model":  model,
        "odoo_field":  field,
        "algo":        "Holt-Winters ETS",
        "n_history":   result["model_info"]["n_points"],
        "periods":     periods,
        "granularity": period,
        "forecast":    forecast_dict,
        "metrics": {
            "mae":      round(metrics.get("mae", 0), 2),
            "rmse":     round(metrics.get("rmse", 0), 2),
            "mape":     metrics.get("mape"),
            "smape":    round(metrics.get("smape", 0), 2),
            "n_points": metrics.get("n_points", 0),
        }
    }, ensure_ascii=False, default=str)


def create_mcp_server() -> FastMCP:
    """Retourne l'instance FastMCP configurée (tools + resources enregistrés au niveau module)."""
    return mcp
