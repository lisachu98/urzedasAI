# urzędasAI — Project-Aware AI Assistant

A self-hosted AI assistant that reads your project folders and answers questions about them.
Supports Claude, GPT-4o, Gemini and Grok— switchable per message.

## Folder Structure

```
your-documents/
├── Project Alpha/
│   ├── brief.pdf
│   ├── notes.docx
│   └── data.xlsx
├── Project Beta/
│   ├── contract.pdf
│   └── images/
└── ...
```

Each subfolder becomes a "project" in the UI.

## Setup

### 1. Install Docker (Container Manager)

### 2. Upload this project

### 3. Set your API keys in .env

### 4. Set your documents path
Edit `docker-compose.yml` and change:
```yaml
- /volume1/documents:/docs:ro
```
to the actual path of your documents
### 5. Build and run
```bash
cd /volume1/docker/urzedasAI
docker compose --env-file .env up -d --build
```

### 6. Access the app
Open `http://localhost:8000` from any browser on the network.

## Adding a New Project
Just create a new folder inside the documents directory and put files in it.

## Supported File Types
| Type | Support |
|------|---------|
| PDF | ✅ Native (text + visual) |
| Images (JPG, PNG, WebP, GIF) | ✅ Native |
| Word (.docx) | ✅ Text extracted |
| Excel (.xlsx) | ✅ Text extracted |
| Plain text, Markdown, CSV, JSON | ✅ Native |
| Code files (.py, .js, .ts, etc.) | ✅ Native |

## Updating
```bash
docker compose --env-file .env up -d --build
```
