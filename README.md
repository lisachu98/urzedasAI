# DocAI — Project-Aware AI Assistant

A self-hosted AI assistant that reads your project folders and answers questions about them.
Supports Claude, GPT-4o, and Gemini — switchable per message.

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

## Setup on Synology NAS

### 1. Install Docker (Container Manager)
Open Package Center → install **Container Manager**.

### 2. Upload this project
Copy the entire `docai/` folder to your NAS (e.g. via File Station to `/volume1/docker/docai`).

### 3. Set your API keys
```bash
cp .env.example .env
# edit .env and add your keys
```
You only need keys for the models you want to use.

### 4. Set your documents path
Edit `docker-compose.yml` and change:
```yaml
- /volume1/documents:/docs:ro
```
to the actual path of your uncle's documents folder on the NAS.

### 5. Build and run
Open Container Manager → Project → Create → point at the docker-compose.yml.

Or via SSH:
```bash
cd /volume1/docker/docai
docker compose --env-file .env up -d --build
```

### 6. Access the app
Open `http://YOUR-NAS-IP:8000` from any browser on the network.

For remote access from anywhere, use Synology's built-in **QuickConnect** or set up a reverse proxy in DSM.

## Adding a New Project
Just create a new folder inside the documents directory and put files in it.
The app will show it automatically — no restart needed.

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
docker compose down
docker compose --env-file .env up -d --build
```
