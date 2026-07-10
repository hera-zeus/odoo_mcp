"""
ST Digital - MCP Server + FastAPI Interface
Architecture: ERP Odoo + Model Context Protocol + Multi-LLM Gateway
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import httpx

from app.config import settings
from app.mcp_server.server import create_mcp_server, init_admin_session
from app.auth.manager import OdooUserAuth
from app.llm_gateway.gateway import LiteLLMGateway
from app.mcp_server.tools import execute_tool, TOOLS_DEFINITION
from app.auth.admin import verify_admin_credentials, create_admin_session, get_admin_session, revoke_admin_session, admin_sessions
from app.forecasting.engine import ForecastEngine
from app.odoo_client.client import OdooClient
import app.mcp_server.resources  # noqa: F401 — enregistre les ressources MCP via les décorateurs

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{settings.LOGS_DIR}/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI(
    title="ST Digital - MCP AI Assistant",
    description="Interface d'aide à la décision augmentée par IA (ERP Odoo + MCP + LLM)",
    version="1.0.0"
)

auth_manager = OdooUserAuth()
odoo_client = OdooClient()  # ← Comme ça, sans paramètres
llm_gateway = LiteLLMGateway(default_model=settings.DEFAULT_LLM)
forecast_engine = ForecastEngine()


# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#@app.middleware("http")
#async def check_api_key(request: Request, call_next):
#    # Routes publiques (pas besoin d'API Key)
#    public_paths = ["/health", "/docs", "/openapi.json", "/auth/login", "/"]
#    
#    if request.url.path not in public_paths:
#        api_key = request.headers.get("X-API-Key")
#        if api_key != settings.API_KEY:
#            raise HTTPException(status_code=401, detail="Invalid or missing API Key")
#    
#    return await call_next(request)


# Initialisation des composants
mcp_server = create_mcp_server()
llm_gateway = LiteLLMGateway(default_model=settings.DEFAULT_LLM)
forecast_engine = ForecastEngine()
# Modèles Pydantic pour les requêtes
class ChatRequest(BaseModel):
    message: str
    llm_model: Optional[str] = settings.DEFAULT_LLM
    session_id: Optional[str] = "default"
class LoginRequest(BaseModel):
    """Modèle pour la requête de connexion"""
    login: str  # Email de l'utilisateur Odoo
    api_key: str  # API Key générée dans le profil Odoo
class ForecastRequestAPI(BaseModel):
    table: str
    field: str
    start_date: str
    end_date: str
    horizon: int = 6
    period: str = "M"  # M=Month, W=Week, D=Day

# ──────────────────────────────────────────────
# Persistance des sessions (fichier JSON)
# ──────────────────────────────────────────────
SESSIONS_FILE = os.path.join(settings.CACHE_DIR, "sessions.json")


def _session_expired(session: dict) -> bool:
    created_at = session.get("created_at")
    if not created_at:
        return True
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    return datetime.now() - created_at > timedelta(seconds=settings.SESSION_TIMEOUT)


def load_sessions() -> Dict[str, dict]:
    """Charge les sessions persistées depuis le fichier JSON (ignore les expirées)."""
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r") as f:
            raw = json.load(f)
        valid = {
            sid: s for sid, s in raw.items()
            if not _session_expired(s)
        }
        expired_count = len(raw) - len(valid)
        if expired_count:
            logger.info(f"{expired_count} session(s) expirée(s) supprimées au chargement")
        # Réhydrater created_at en datetime
        for s in valid.values():
            s["created_at"] = datetime.fromisoformat(s["created_at"])
            s.setdefault("history", [])
        return valid
    except Exception as e:
        logger.error(f"Erreur chargement sessions: {e}")
        return {}


def save_sessions(sessions: Dict[str, dict]) -> None:
    """Persiste les sessions actives dans le fichier JSON (sans l'historique de chat)."""
    try:
        serializable = {}
        for sid, s in sessions.items():
            serializable[sid] = {
                "user":             s["user"],
                "odoo_session_id":  s["odoo_session_id"],
                "created_at":       s["created_at"].isoformat()
                                    if isinstance(s["created_at"], datetime)
                                    else s["created_at"],
                # history non persisté : trop volumineux et lié à la conversation
            }
        with open(SESSIONS_FILE, "w") as f:
            json.dump(serializable, f)
    except Exception as e:
        logger.error(f"Erreur sauvegarde sessions: {e}")


sessions: Dict[str, dict] = load_sessions()
logger.info(f"{len(sessions)} session(s) restaurée(s) depuis {SESSIONS_FILE}")
# ==========================================
# Catalogue des Tools MCP (Function Calling)
# ==========================================
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_odoo_records",
            "description": (
                "Recherche et récupère des données depuis Odoo sur n'importe quel domaine "
                "(Ventes, Achats, Budget, Contacts, Stocks, Comptabilité). "
                "Utilise cet outil pour répondre aux questions factuelles sur les données de l'entreprise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Le modèle Odoo à interroger (ex: 'sale.order', 'account.move', 'res.partner', 'purchase.order', 'product.product', 'account.budget.post')"
                    },
                    "domain": {
                        "type": "array",
                        "items": {"type": "array"},
                        "description": "Les filtres au format Odoo (ex: [['state', '=', 'sale'], ['date_order', '>=', '2026-01-01']])"
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Les champs à récupérer (ex: ['name', 'amount_total', 'date_order'])"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de résultats (défaut: 50)"
                    }
                },
                "required": ["model", "domain", "fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_forecast",
            "description": (
                "Génère une prévision à 6 mois basée sur l'historique d'un champ numérique "
                "d'un modèle Odoo. Utilise le lissage exponentiel (ETS). "
                "À utiliser quand l'utilisateur demande une prévision, une projection ou une tendance future."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Modèle source (ex: 'sale.order', 'account.move')"
                    },
                    "field": {
                        "type": "string",
                        "description": "Champ numérique à prévoir (ex: 'amount_total', 'quantity')"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Date de début de l'historique (format: YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Date de fin de l'historique (format: YYYY-MM-DD)"
                    },
                    "period": {
                        "type": "string",
                        "description": "Granularité temporelle : 'D' (jour), 'W' (semaine), 'M' (mois), 'Q' (trimestre). Défaut: 'M'"
                    }
                },
                "required": ["model", "field", "start_date", "end_date"]
            }
        }
    }
]


@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    logger.info("Démarrage de l'application ST Digital MCP")
    logger.info(f"Odoo URL: {settings.ODOO_URL}")
    logger.info(f"LLM par défaut: {settings.DEFAULT_LLM}")

    # Initialiser la session admin pour le serveur MCP
    await init_admin_session()

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    logger.info("🛑 Arrêt de l'application")
    await odoo_client.close()

@app.get("/health")
async def health_check():
    """Endpoint de santé du système"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "mcp_server": "running",
            "llm_gateway": "running",
            "forecast_engine": "running",
            "odoo_client": "running"
        }
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil simple"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ST Digital - AI Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🚀 ST Digital - AI Assistant</h1>
        <p>Interface d'aide à la décision augmentée par IA</p>
        
        <h2>Endpoints disponibles:</h2>
        <div class="endpoint">
            <strong>GET</strong> <code>/health</code> - Santé du système
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/chat</code> - Chat conversationnel
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/api/forecast</code> - Prévisions
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/api/sales</code> - Données de ventes
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/docs</code> - Documentation Swagger
        </div>
        
        <p><em>Documentation complète: <a href="/docs">/docs</a></em></p>
    </body>
    </html>
    """

MAX_TOOL_RESULT_CHARS = 6000   # ~1500 tokens — évite d'exploser la limite 10k/min


def truncate_tool_result(content: str) -> str:
    """Tronque un résultat d'outil trop long pour rester sous la limite de tokens."""
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    truncated = content[:MAX_TOOL_RESULT_CHARS]
    # Couper proprement à la dernière virgule JSON pour ne pas casser le parsing
    last_comma = truncated.rfind(',')
    if last_comma > MAX_TOOL_RESULT_CHARS * 0.8:
        truncated = truncated[:last_comma]
    return truncated + '\n... [résultat tronqué — trop volumineux]'


def safe_history_slice(history: list, max_messages: int = 10) -> list:
    """
    Retourne jusqu'à max_messages messages sans jamais couper une séquence
    tool_call/tool_result. Commence toujours par un message 'user'.
    """
    if not history:
        return []
    if len(history) <= max_messages:
        return history

    candidate = history[-max_messages:]

    # Reculer jusqu'au premier message 'user' pour ne pas commencer
    # par un 'tool' ou 'assistant' orphelin de son tool_use
    for i, msg in enumerate(candidate):
        if msg.get("role") == "user":
            return candidate[i:]

    # Fallback : dernier message user uniquement
    for msg in reversed(history):
        if msg.get("role") == "user":
            return [msg]
    return []


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Gère les requêtes conversationnelles avec appel dynamique d'outils MCP.
    Le LLM peut appeler les outils search_odoo_records et generate_forecast
    pour répondre aux questions de l'utilisateur.
    """
    # Vérification et TTL de la session
    if request.session_id not in sessions:
        raise HTTPException(status_code=401, detail="Session invalide")

    session_data = sessions[request.session_id]
    if _session_expired(session_data):
        del sessions[request.session_id]
        save_sessions(sessions)
        raise HTTPException(status_code=401, detail="Session expirée")

    # Renouveler le TTL à chaque activité
    session_data["created_at"] = datetime.now()

    user            = session_data["user"]
    odoo_session_id = session_data["odoo_session_id"]

    # Récupération de l'historique depuis la session
    session_history = session_data.get("history", [])
    
    logger.info(f"💬 Nouvelle requête chat de {user['name']}: {request.message[:80]}...")
    
    # 1. Ajout du message utilisateur à l'historique
    session_history.append({"role": "user", "content": request.message})
    
    # 2. Prompt système contextualisé
    today = datetime.now()
    system_prompt = f"""
    Tu es l'assistant IA d'aide à la décision de {settings.COMPANY_NAME}.
    Utilisateur actuel : {user['name']} ({user['email']}).
    Date et heure actuelles : {today.strftime("%A %d %B %Y, %H:%M")} (fuseau serveur).
    Année en cours : {today.year}. Mois en cours : {today.month}.

    RÈGLES STRICTES :
    - Réponds toujours en français, de manière professionnelle et concise.
    - Base tes réponses UNIQUEMENT sur les données retournées par les outils MCP.
    - N'invente JAMAIS de chiffres (pas d'hallucinations).
    - Si tu n'as pas les données ou si l'outil retourne une erreur, dis-le clairement.
    - Respecte la confidentialité : ne donne pas d'informations sur les données d'autres utilisateurs.
    - La devise de l'entreprise est le {settings.COMPANY_CURRENCY}. Utilise TOUJOURS cette devise pour les montants. N'utilise jamais €, $, ou toute autre devise sauf si explicitement demandé.
    - Pour les prévisions, mentionne toujours la marge d'erreur (MAPE ou sMAPE).
    - Pour toute requête sans date précisée, utilise l'année en cours ({today.year}) comme référence.
    """

    try:
        # 3. Construction du contexte pour le LLM
        messages = [
            {"role": "system", "content": system_prompt},
            *safe_history_slice(session_history)
        ]
        
        # 4. Appel initial au LLM avec les Tools disponibles
        response = await llm_gateway.chat_completion_with_tools(
            messages=messages,
            tools=TOOLS_DEFINITION,
            model=request.llm_model
        )

        # 5. Boucle de gestion des Tool Calls
        max_tool_iterations = 5  # Sécurité contre les boucles infinies
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_tool_iterations:
            iteration += 1
            tool_calls = response.choices[0].message.tool_calls
            
            # On ajoute la réponse du LLM (qui contient l'appel d'outil) à l'historique
            session_history.append({
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # On exécute chaque outil demandé par le LLM
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"🛠️ Exécution de l'outil : {function_name} par {user['name']}")
                
                # Exécution via le routeur d'outils (avec la session Odoo de l'utilisateur)
                function_response = await execute_tool(
                    tool_name=function_name,
                    tool_args=function_args,
                    odoo_session_id=odoo_session_id
                )

                # On injecte le résultat de l'outil dans l'historique pour le LLM
                # (tronqué pour ne pas dépasser la limite de tokens)
                session_history.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": truncate_tool_result(function_response)
                })

            # 6. On renvoie l'historique enrichi au LLM pour qu'il génère la réponse finale
            messages = [
                {"role": "system", "content": system_prompt},
                *safe_history_slice(session_history)
            ]

            response = await llm_gateway.chat_completion_with_tools(
                messages=messages,
                tools=TOOLS_DEFINITION,
                model=request.llm_model
            )

        # 7. Réponse finale textuelle
        final_answer = response.choices[0].message.content
        session_history.append({"role": "assistant", "content": final_answer})
        
        # 8. Sauvegarde de l'historique dans la session (SANS écraser les autres données)
        sessions[request.session_id]["history"] = session_history[-10:]  # Garde max 20 messages
        
        logger.info(f"✅ Réponse générée avec succès pour {user['name']} ({len(final_answer)} caractères)")

        return {
            "response": final_answer,
            "session_id": request.session_id,
            "model_used": request.llm_model,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Erreur dans le flux MCP : {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du traitement de la requête : {str(e)}"
        )

@app.post("/api/forecast")
async def forecast_endpoint(request: ForecastRequestAPI):
    """
    Endpoint de prévision utilisant le Moteur Prédictif Universel
    """
    try:
        logger.info(f"📈 Nouvelle prévision: {request.table}.{request.field}")
        
        # 1. Récupérer les données historiques depuis Odoo
        data = await odoo_client.get_time_series(
            table=request.table,
            field=request.field,
            start_date=request.start_date,
            end_date=request.end_date,
            period=request.period
        )
        
        if not data or len(data) < 12:
            raise HTTPException(
                status_code=400,
                detail=f"Données insuffisantes: {len(data)} points (minimum 12 requis)"
            )
        
        # 2. Générer la prévision
        forecast_result = forecast_engine.forecast_ets(
            series=data,
            periods=request.horizon
        )
        
        # 3. Calculer les métriques sur la série préprocessée (cohérente avec le modèle)
        metrics = forecast_engine.calculate_metrics(
            actual=forecast_result['series_clean'],
            predicted=forecast_result['fitted']
        )

        logger.info(f"Prévision générée: MAPE={metrics['mape']}%, sMAPE={metrics['smape']}%")
        
        return {
            "forecast": forecast_result['forecast'].tolist(),
            "fitted_values": forecast_result['fitted'].tolist(),
            "metrics": metrics,
            "dates": {
                "historical": [str(d) for d in data.index],
                "future": [str(d) for d in forecast_result['forecast_dates']]
            },
            "model_info": forecast_result['model_info'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur prévision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sales")
async def get_sales_data(
    start_date: str,
    end_date: str,
    category: Optional[str] = None
):
    """
    Récupérer les données de ventes depuis Odoo
    """
    try:
        logger.info(f"📊 Récupération des ventes: {start_date} -> {end_date}")
        
        sales_data = await odoo_client.get_sales_data(
            start_date=start_date,
            end_date=end_date,
            category=category
        )
        
        return {
            "data": sales_data,
            "count": len(sales_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération ventes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class AdminLoginRequest(BaseModel):
    username: str
    password: str

class LLMConfigRequest(BaseModel):
    default_llm: str


def require_admin(request: Request) -> dict:
    """Dépendance FastAPI — vérifie qu'une session admin valide est fournie"""
    admin_session_id = request.headers.get("X-Admin-Session-ID")
    if not admin_session_id:
        raise HTTPException(status_code=401, detail="Session admin requise")
    admin = get_admin_session(admin_session_id)
    if not admin:
        raise HTTPException(status_code=401, detail="Session admin invalide ou expirée")
    return admin


@app.post("/auth/admin-login")
async def admin_login_endpoint(credentials: AdminLoginRequest):
    """Authentification du Super Admin (indépendante d'Odoo)"""
    logger.info(f"🔐 Tentative de connexion admin: {credentials.username}")

    if not verify_admin_credentials(credentials.username, credentials.password):
        logger.warning(f"❌ Échec authentification admin: {credentials.username}")
        raise HTTPException(status_code=401, detail="Identifiants administrateur invalides")

    admin_session_id = create_admin_session(credentials.username)
    return {
        "admin_session_id": admin_session_id,
        "username":         credentials.username,
        "role":             "super_admin"
    }


@app.post("/admin/logout")
async def admin_logout_endpoint(request: Request):
    admin_session_id = request.headers.get("X-Admin-Session-ID")
    revoke_admin_session(admin_session_id)
    return {"status": "déconnecté"}


@app.get("/admin/sessions")
async def list_sessions(admin: dict = Depends(require_admin)):
    """Liste toutes les sessions utilisateurs actives"""
    result = []
    for sid, data in sessions.items():
        result.append({
            "session_id": sid,
            "user":       data.get("user", {}).get("name", "Inconnu"),
            "email":      data.get("user", {}).get("email", ""),
            "messages":   len(data.get("history", []))
        })
    return {"count": len(result), "sessions": result}


@app.post("/admin/sessions/{session_id}/revoke")
async def revoke_session(session_id: str, admin: dict = Depends(require_admin)):
    """Révoque une session utilisateur"""
    if session_id in sessions:
        del sessions[session_id]
        save_sessions(sessions)
        logger.info(f"Session révoquée par admin: {session_id}")
        return {"status": "révoquée", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session introuvable")


@app.get("/admin/config")
async def get_config(admin: dict = Depends(require_admin)):
    """Retourne la configuration actuelle du système"""
    return {
        "default_llm":       settings.DEFAULT_LLM,
        "odoo_url":          settings.ODOO_URL,
        "odoo_db":           settings.ODOO_DB,
        "active_sessions":   len(sessions),
        "llms_disponibles":  [
            "anthropic/claude-sonnet-4-6",
            "openai/gpt-4o-mini",
            "gemini/gemini-2.0-flash"
        ]
    }


@app.post("/admin/config/llm")
async def update_default_llm(config: LLMConfigRequest, admin: dict = Depends(require_admin)):
    """Change le LLM par défaut (en mémoire, pour la session serveur en cours)"""
    settings.DEFAULT_LLM = config.default_llm
    logger.info(f"⚙️ LLM par défaut changé en {config.default_llm} par {admin['username']}")
    return {"status": "mis à jour", "default_llm": settings.DEFAULT_LLM}


@app.post("/auth/login")
async def login_endpoint(credentials: LoginRequest):
    """Authentifie l'utilisateur auprès d'Odoo via son Login et son API Key."""
    logger.info(f"Tentative de connexion pour l'utilisateur: {credentials.login}")
    
    user_info = await auth_manager.authenticate_user(credentials.login, credentials.api_key)
    
    if not user_info:
        raise HTTPException(
            status_code=401, 
            detail="Échec de l'authentification. Vérifiez votre login et votre API Key Odoo."
        )
    
    # Création et persistance de la session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user":             user_info,
        "odoo_session_id":  user_info["odoo_session_id"],
        "history":          [],
        "created_at":       datetime.now()
    }
    save_sessions(sessions)
    logger.info(f"Connexion réussie pour {user_info['name']} (Session: {session_id})")
    
    return {
        "session_id": session_id,
        "user": {
            "uid": user_info["uid"],
            "name": user_info["name"],
            "email": user_info["email"]
        }
    }


@app.get("/api/session/check")
async def check_session(request: Request):
    """Vérifie si une session utilisateur est encore valide et non expirée"""
    session_id = request.headers.get("X-Session-ID")
    if session_id and session_id in sessions:
        session = sessions[session_id]
        if _session_expired(session):
            del sessions[session_id]
            save_sessions(sessions)
            raise HTTPException(status_code=401, detail="Session expirée")
        user = session["user"]
        return {"valid": True, "user": {"uid": user["uid"], "name": user["name"], "email": user["email"]}}
    raise HTTPException(status_code=401, detail="Session invalide ou expirée")


@app.get("/api/llms")
async def list_llms():
    """Liste des LLMs disponibles via LiteLLM"""
    return {
        "available_llms": [
            {"id": "anthropic/claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "provider": "Anthropic"},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
            {"id": "gemini/gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "Google"}
        ],
        "default": settings.DEFAULT_LLM
    }

async def build_mcp_context() -> Dict[str, Any]:
    """
    Construire le contexte MCP avec les données Odoo récentes
    """
    context = {
        "available_resources": [
            "odoo://sales/orders",
            "odoo://sales/order_lines",
            "odoo://partners",
            "odoo://accounting/moves",
            "odoo://products"
        ],
        "available_tools": [
            "get_sales_data",
            "get_customer_info",
            "forecast_sales",
            "analyze_trends"
        ],
        "current_date": datetime.now().isoformat()
    }
    return context

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="app/web_interface/static"), name="static")

# Monter le serveur MCP (transport SSE — compatible Claude Desktop et clients MCP)
app.mount("/mcp", mcp_server.http_app(transport="sse"), name="mcp")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.ENV == "development"),
        workers=2 if settings.ENV == "production" else 1
    )
