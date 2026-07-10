"""
Ressources MCP exposant des données Odoo en lecture.
Les ressources sont des URIs navigables par les clients MCP (ex: Claude Desktop).
"""

import json
from datetime import datetime, timedelta
from app.mcp_server.server import mcp, _odoo, _admin_session


# ──────────────────────────────────────────────
# VENTES
# ──────────────────────────────────────────────

@mcp.resource("odoo://sales/recent")
async def recent_sales() -> str:
    """Commandes de ventes confirmées des 30 derniers jours"""
    since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    records = await _odoo.search_read(
        model="sale.order",
        domain=[["state", "in", ["sale", "done"]], ["date_order", ">=", since]],
        fields=["name", "partner_id", "amount_total", "date_order", "state"],
        odoo_session_id=_admin_session,
        limit=50
    )
    return json.dumps({"count": len(records), "since": since, "orders": records},
                      ensure_ascii=False, default=str)


@mcp.resource("odoo://sales/top-customers")
async def top_customers() -> str:
    """Top 20 clients par montant total de commandes (année en cours)"""
    year_start = f"{datetime.now().year}-01-01"
    records = await _odoo.search_read(
        model="sale.order",
        domain=[["state", "in", ["sale", "done"]], ["date_order", ">=", year_start]],
        fields=["partner_id", "amount_total"],
        odoo_session_id=_admin_session,
        limit=5000
    )

    # Agréger par client
    totals: dict = {}
    for r in records:
        partner = r.get("partner_id")
        if not partner:
            continue
        pid, pname = (partner[0], partner[1]) if isinstance(partner, list) else (partner, str(partner))
        totals[pid] = {"name": pname, "total": totals.get(pid, {}).get("total", 0) + r.get("amount_total", 0)}

    ranked = sorted(totals.values(), key=lambda x: x["total"], reverse=True)[:20]
    return json.dumps({"year": datetime.now().year, "top_customers": ranked},
                      ensure_ascii=False, default=str)


# ──────────────────────────────────────────────
# COMPTABILITÉ
# ──────────────────────────────────────────────

@mcp.resource("odoo://accounting/unpaid-invoices")
async def unpaid_invoices() -> str:
    """Factures clients non payées (move_type=out_invoice, payment_state!=paid)"""
    records = await _odoo.search_read(
        model="account.move",
        domain=[
            ["move_type", "=", "out_invoice"],
            ["state", "=", "posted"],
            ["payment_state", "in", ["not_paid", "partial"]]
        ],
        fields=["name", "partner_id", "amount_residual", "invoice_date_due", "payment_state"],
        odoo_session_id=_admin_session,
        limit=100
    )
    total_due = sum(r.get("amount_residual", 0) for r in records)
    return json.dumps({"count": len(records), "total_due": round(total_due, 2), "invoices": records},
                      ensure_ascii=False, default=str)


# ──────────────────────────────────────────────
# RH
# ──────────────────────────────────────────────

@mcp.resource("odoo://hr/employees")
async def employees() -> str:
    """Liste des employés actifs"""
    records = await _odoo.search_read(
        model="hr.employee",
        domain=[["active", "=", True]],
        fields=["name", "job_title", "department_id", "work_email", "coach_id"],
        odoo_session_id=_admin_session,
        limit=200
    )
    return json.dumps({"count": len(records), "employees": records},
                      ensure_ascii=False, default=str)


# ──────────────────────────────────────────────
# PRODUITS
# ──────────────────────────────────────────────

@mcp.resource("odoo://products/catalog")
async def product_catalog() -> str:
    """Catalogue des produits actifs avec prix de vente"""
    records = await _odoo.search_read(
        model="product.template",
        domain=[["active", "=", True], ["sale_ok", "=", True]],
        fields=["name", "list_price", "categ_id", "type", "default_code"],
        odoo_session_id=_admin_session,
        limit=200
    )
    return json.dumps({"count": len(records), "products": records},
                      ensure_ascii=False, default=str)
