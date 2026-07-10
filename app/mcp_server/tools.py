import json
import logging
from typing import Any, Dict
from app.odoo_client.client import OdooClient

logger = logging.getLogger(__name__)

odoo_client = OdooClient()

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_odoo_records",
            "description": "Recherche et retourne des enregistrements depuis n'importe quel modèle Odoo. Utilise cet outil pour répondre à des questions sur les ventes, budgets, contacts, achats, employés, factures, stocks, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Nom technique du modèle Odoo (ex: sale.order, account.move, res.partner, purchase.order, hr.employee, stock.quant, account.analytic.account)"
                    },
                    "domain": {
                        "type": "array",
                        "description": "Filtre de recherche Odoo (ex: [['state','=','sale']] pour ventes confirmées, [['date_order','>=','2026-01-01']] pour filtrer par date)",
                        "default": []
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des champs à retourner (ex: ['name','amount_total','partner_id','state','date_order'])"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum d'enregistrements à retourner",
                        "default": 20
                    }
                },
                "required": ["model", "fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Génère des prévisions sur les prochains mois pour n'importe quel champ numérique "
                "d'un modèle Odoo, via lissage exponentiel Holt-Winters (ETS). "
                "Utilise cet outil quand l'utilisateur demande des prévisions, projections ou tendances futures "
                "(ventes, achats, charges salariales, consommation stock, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Modèle Odoo source (ex: sale.order, account.move, purchase.order, hr.payslip, stock.move)"
                    },
                    "field": {
                        "type": "string",
                        "description": "Champ numérique à prévoir (ex: amount_total, net_wage, product_uom_qty)"
                    },
                    "date_field": {
                        "type": "string",
                        "description": "Champ date du modèle (ex: date_order pour sale.order, invoice_date pour account.move, date_from pour hr.payslip)"
                    },
                    "domain": {
                        "type": "array",
                        "description": "Filtres additionnels Odoo (ex: [['state','=','posted']] pour factures validées)",
                        "default": []
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Début de l'historique (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Fin de l'historique (YYYY-MM-DD)"
                    },
                    "periods": {
                        "type": "integer",
                        "description": "Nombre de périodes à prévoir (défaut: 6)",
                        "default": 6
                    },
                    "period": {
                        "type": "string",
                        "description": "Granularité : 'D' (jour), 'W' (semaine), 'M' (mois), 'Q' (trimestre). Défaut: 'M'",
                        "default": "M"
                    }
                },
                "required": ["model", "field", "date_field", "start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_odoo_models",
            "description": "Retourne la liste des modèles Odoo disponibles. Utilise cet outil quand tu ne sais pas quel modèle utiliser pour répondre à une question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Mot-clé pour filtrer les modèles (ex: 'sale', 'account', 'hr')"
                    }
                },
                "required": []
            }
        }
    }
]


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    odoo_session_id: str
) -> str:
    """Routeur principal — exécute l'outil MCP demandé par le LLM"""
    try:
        logger.info(f"🔧 execute_tool: {tool_name} avec args={tool_args}")

        if tool_name == "search_odoo_records":
            return await _search_odoo_records(tool_args, odoo_session_id)

        elif tool_name == "get_forecast":
            return await _get_forecast(tool_args, odoo_session_id)

        elif tool_name == "get_odoo_models":
            return await _get_odoo_models(tool_args, odoo_session_id)

        else:
            return json.dumps({"error": f"Outil inconnu : {tool_name}"})

    except Exception as e:
        logger.error(f"❌ Erreur execute_tool [{tool_name}]: {e}")
        return json.dumps({"error": str(e)})


async def _search_odoo_records(args: Dict, odoo_session_id: str) -> str:
    """Recherche des enregistrements dans un modèle Odoo"""
    model  = args.get("model", "sale.order")
    domain = args.get("domain", [])
    fields = args.get("fields", ["name"])
    limit  = min(args.get("limit", 10), 50)   # max 50 pour rester sous la limite de tokens

    records = await odoo_client.search_read(
        model=model,
        domain=domain,
        fields=fields,
        odoo_session_id=odoo_session_id,
        limit=limit
    )

    if not records:
        return json.dumps({
            "model": model,
            "count": 0,
            "message": f"Aucun enregistrement trouvé dans {model} avec ce filtre.",
            "records": []
        }, ensure_ascii=False)

    return json.dumps({
        "model":   model,
        "count":   len(records),
        "records": records
    }, ensure_ascii=False, default=str)


async def _get_forecast(args: Dict, odoo_session_id: str) -> str:
    """Génère des prévisions sur n'importe quel champ numérique d'un modèle Odoo"""
    try:
        from app.forecasting.engine import ForecastEngine
        engine     = ForecastEngine()
        model      = args.get("model", "sale.order")
        field      = args.get("field", "amount_total")
        date_field = args.get("date_field", "date_order")
        domain     = args.get("domain", [])
        start_date = args.get("start_date", "2024-01-01")
        end_date   = args.get("end_date", "2026-12-31")
        periods    = args.get("periods", 6)
        period     = args.get("period", "M")

        series = await odoo_client.get_time_series(
            table=model,
            field=field,
            date_field=date_field,
            domain=domain,
            start_date=start_date,
            end_date=end_date,
            period=period,
            odoo_session_id=odoo_session_id
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

    except Exception as e:
        return json.dumps({"error": f"Erreur prévision : {str(e)}"})


async def _get_odoo_models(args: Dict, odoo_session_id: str) -> str:
    """Retourne la liste des modèles Odoo disponibles"""
    keyword = args.get("keyword", "")

    records = await odoo_client.search_read(
        model="ir.model",
        domain=[["model", "like", keyword]] if keyword else [],
        fields=["model", "name"],
        odoo_session_id=odoo_session_id,
        limit=50
    )

    return json.dumps({
        "count":  len(records),
        "models": records
    }, ensure_ascii=False)
