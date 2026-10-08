# Installation

## 1. L'environnement Python (avec uv)

```bash
uv init cours-rag --python 3.12
cd cours-rag
uv add "docling>=2.94" pymupdf transformers polars "psycopg[binary]" qdrant-client openai jupyterlab
```

## 2. Les services (avec Docker)

PostgreSQL sert de journal de bord et Qdrant de base vectorielle. Créez un fichier `docker-compose.yml` dans le dossier du cours :

Les commandes utiles, à lancer depuis le dossier qui contient `docker-compose.yml` :

| Commande | Effet |
|---|---|
| `docker compose up -d` | Démarre les services en arrière-plan |
| `docker compose ps` | Vérifie qu'ils tournent |
| `docker compose logs postgres` | Affiche les messages de PostgreSQL |
| `docker compose stop` | Arrête les services (les données sont conservées) |
| `docker compose down -v` | Supprime tout, **données comprises** |

Pour vérifier que tout fonctionne :

```bash
# PostgreSQL : ouvre une console SQL (quitter avec \q)
docker compose exec postgres psql -U cours -d coursrag

# Qdrant : doit afficher un petit texte au format JSON avec le numéro de version
curl http://localhost:6333
```

Qdrant propose aussi un tableau de bord dans le navigateur : http://localhost:6333/dashboard

## 3. Les modèles

Tout est déjà le dossier /models mais vous pouvez aussi les télécharger avec la commande suivante:

```bash
uv run docling-tools models download -o models/docling
 
uv run hf download BAAI/bge-m3 --local-dir models/embedder/bge-m3 --exclude "onnx/*"
 
uv run hf download BAAI/bge-reranker-v2-m3 --local-dir models/reranker/bge-reranker-v2-m3
```