#!/bin/bash

# Activer l'environnement virtuel
source /home/mcp/venv/bin/activate

# Aller dans le dossier du projet
cd /home/mcp/std_mcp

# Démarrer l'application avec Uvicorn
echo "🚀 Démarrage de ST Digital MCP Server..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2 --reload
