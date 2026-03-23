import os
import base64
import mimetypes
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import anthropic
import openai
from google import genai
from google.genai import types as gtypes
import fitz
import re
import time
from functools import lru_cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_ROOT = os.environ.get("DOCS_ROOT", "/docs")

# ── singleton API clients ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)

@lru_cache(maxsize=1)
def get_openai_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

@lru_cache(maxsize=1)
def get_xai_client() -> openai.OpenAI:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="XAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

# ── model listing (cached) ───────────────────────────────────────────────────

_models_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 3600  # 1 hour
_OAI_DATED_RE = re.compile(r"\d{4,}")  # filters out dated snapshots like gpt-4o-2024-08-06


def fetch_available_models() -> dict:
    """Query each provider for available models. Results cached for 1 hour."""
    now = time.time()
    if _models_cache["data"] and (now - _models_cache["ts"]) < _CACHE_TTL:
        return _models_cache["data"]

    result: dict[str, list] = {}

    # ── Anthropic ──
    try:
        client = get_anthropic_client()
        page = client.models.list(limit=100)
        result["anthropic"] = sorted(
            [{"id": m.id, "name": m.display_name} for m in page.data],
            key=lambda x: x["name"],
        )
    except Exception:
        result["anthropic"] = []

    # ── OpenAI ──
    try:
        client = get_openai_client()
        oai: list[dict] = []
        for m in client.models.list().data:
            mid = m.id
            # only chat-capable model families
            if not (mid.startswith("gpt-") or re.match(r"^o\d", mid) or mid.startswith("chatgpt")):
                continue
            # skip non-chat variants
            if any(kw in mid for kw in ("realtime", "audio", "transcribe", "instruct", "search")):
                continue
            # skip dated snapshots (e.g. gpt-4o-2024-08-06)
            if _OAI_DATED_RE.search(mid):
                continue
            oai.append({"id": mid, "name": mid})
        result["openai"] = sorted(oai, key=lambda x: x["name"])
    except Exception:
        result["openai"] = []

    # ── Gemini ──
    try:
        client = get_gemini_client()
        gem: list[dict] = []
        for m in client.models.list():
            model_id = m.name.replace("models/", "") if m.name else ""
            if not model_id.startswith("gemini"):
                continue
            gem.append({"id": model_id, "name": m.display_name or model_id})
        result["gemini"] = sorted(gem, key=lambda x: x["name"])
    except Exception:
        result["gemini"] = []

    _models_cache["data"] = result
    _models_cache["ts"] = now
    return result

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    text_parts = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
        doc.close()
    except Exception:
        return "[Could not extract text from PDF]"
    return "\n".join(text_parts) if text_parts else "[PDF contained no extractable text]"


def list_projects():
    root = Path(DOCS_ROOT)
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def load_folder(folder_path: Path):
    """Return a list of content blocks (text or image/pdf) for all files in folder."""
    SUPPORTED_TEXT  = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".py",
                       ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs"}
    SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    SUPPORTED_DIRECT = {".pdf"}

    # docx / xlsx need extraction
    try:
        import docx as _docx
        HAS_DOCX = True
    except ImportError:
        HAS_DOCX = False
    try:
        import openpyxl as _openpyxl
        HAS_XLSX = True
    except ImportError:
        HAS_XLSX = False

    blocks = []
    files  = sorted(folder_path.rglob("*"))

    for f in files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()

        # ── plain text ──
        if ext in SUPPORTED_TEXT:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                blocks.append({"type": "text", "text": f"--- FILE: {f.name} ---\n{content}\n"})
            except Exception:
                pass

        # ── images ──
        elif ext in SUPPORTED_IMAGE:
            try:
                data    = base64.standard_b64encode(f.read_bytes()).decode()
                mime, _ = mimetypes.guess_type(str(f))
                mime    = mime or "image/jpeg"
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                blocks.append({"type": "text", "text": f"(image above: {f.name})\n"})
            except Exception:
                pass

        # ── PDF ──
        elif ext in SUPPORTED_DIRECT:
            try:
                raw  = f.read_bytes()
                data = base64.standard_b64encode(raw).decode()
                blocks.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                    "_pdf_text": extract_pdf_text(raw),   # pre-extract for models that need it
                    "_filename": f.name,
                })
                blocks.append({"type": "text", "text": f"(document above: {f.name})\n"})
            except Exception:
                pass

        # ── Word ──
        elif ext == ".docx" and HAS_DOCX:
            try:
                import docx
                doc  = docx.Document(str(f))
                text = "\n".join(p.text for p in doc.paragraphs)
                blocks.append({"type": "text", "text": f"--- FILE: {f.name} ---\n{text}\n"})
            except Exception:
                pass

        # ── Excel ──
        elif ext in (".xlsx", ".xls") and HAS_XLSX:
            try:
                import openpyxl
                wb   = openpyxl.load_workbook(str(f), data_only=True)
                text = ""
                for sheet in wb.sheetnames:
                    ws   = wb[sheet]
                    text += f"[Sheet: {sheet}]\n"
                    for row in ws.iter_rows(values_only=True):
                        text += "\t".join(str(c) if c is not None else "" for c in row) + "\n"
                blocks.append({"type": "text", "text": f"--- FILE: {f.name} ---\n{text}\n"})
            except Exception:
                pass

    return blocks


# ── request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    project:  Optional[str] = None
    question: str
    provider: str  # "anthropic" | "openai" | "gemini"
    model:    str  # actual model ID, e.g. "claude-sonnet-4-5"
    history:  Optional[list] = []


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def get_projects():
    return {"projects": list_projects()}


@app.get("/api/models")
def get_models():
    return fetch_available_models()


@app.post("/api/chat")
def chat(req: ChatRequest):
    # load project files if one is selected
    context_blocks = []
    if req.project:
        folder = (Path(DOCS_ROOT) / req.project).resolve()
        # ── path traversal guard ──
        if not str(folder).startswith(str(Path(DOCS_ROOT).resolve())):
            raise HTTPException(status_code=400, detail="Invalid project path")
        if not folder.exists():
            raise HTTPException(status_code=404, detail="Project folder not found")
        context_blocks = load_folder(folder)

    # build user message
    user_content = context_blocks + [{"type": "text", "text": req.question}]

    system_prompt = (
        "You are a helpful assistant with access to the user's project documents. "
        "Answer questions based on the provided documents. "
        "When asked to write something similar to existing documents, match their style, "
        "tone, structure, and language. Be precise and professional."
    )

    # ── Claude ───────────────────────────────────────────────────────────────
    if req.provider == "anthropic":
        client = get_anthropic_client()
        # Strip internal keys from blocks before sending to Anthropic
        clean_content = []
        for block in user_content:
            b = {k: v for k, v in block.items() if not k.startswith("_")}
            clean_content.append(b)
        messages = (req.history or []) + [{"role": "user", "content": clean_content}]
        response = client.messages.create(
            model      = req.model,
            max_tokens = 4096,
            system     = system_prompt,
            messages   = messages,
        )
        return {"answer": response.content[0].text}

    # ── GPT-4o ───────────────────────────────────────────────────────────────
    elif req.provider == "openai":
        client = get_openai_client()

        # convert blocks to OpenAI format
        def to_oai(block):
            if block["type"] == "text":
                return {"type": "text", "text": block["text"]}
            if block["type"] == "image":
                src = block["source"]
                return {"type": "image_url", "image_url": {
                    "url": f"data:{src['media_type']};base64,{src['data']}"
                }}
            if block["type"] == "document":
                # GPT-4o can't read raw PDFs — use pre-extracted text
                pdf_text = block.get("_pdf_text", "[PDF content unavailable]")
                filename = block.get("_filename", "document.pdf")
                return {"type": "text", "text": f"--- FILE: {filename} ---\n{pdf_text}\n"}
            return {"type": "text", "text": ""}

        oai_content = [to_oai(b) for b in user_content]
        messages    = [{"role": "system", "content": system_prompt}]
        for h in (req.history or []):
            messages.append(h)
        messages.append({"role": "user", "content": oai_content})

        response = client.chat.completions.create(model=req.model, messages=messages, max_tokens=4096)
        return {"answer": response.choices[0].message.content}

    # ── Gemini ────────────────────────────────────────────────────────────────
    elif req.provider == "gemini":
        client = get_gemini_client()

        # Build parts for the current message
        parts: list = []
        for block in user_content:
            if block["type"] == "text":
                parts.append(gtypes.Part.from_text(text=block["text"]))
            elif block["type"] == "image":
                src = block["source"]
                parts.append(gtypes.Part.from_bytes(
                    data=base64.b64decode(src["data"]),
                    mime_type=src["media_type"],
                ))
            elif block["type"] == "document":
                parts.append(gtypes.Part.from_bytes(
                    data=base64.b64decode(block["source"]["data"]),
                    mime_type="application/pdf",
                ))

        config = gtypes.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
        )

        # Build conversation history for Gemini
        gemini_history: list = []
        for h in (req.history or []):
            role = "user" if h["role"] == "user" else "model"
            gemini_history.append(gtypes.Content(
                role=role,
                parts=[gtypes.Part.from_text(text=h["content"])],
            ))

        if gemini_history:
            chat_session = client.chats.create(
                model=req.model,
                config=config,
                history=gemini_history,
            )
            response = chat_session.send_message(parts)
        else:
            response = client.models.generate_content(
                model=req.model,
                contents=parts,
                config=config,
            )

        return {"answer": response.text}

    # ── xAI (Grok) ────────────────────────────────────────────────────────────
    elif req.provider == "xai":
        client = get_xai_client()

        # xAI uses OpenAI-compatible format
        def to_xai(block):
            if block["type"] == "text":
                return {"type": "text", "text": block["text"]}
            if block["type"] == "image":
                src = block["source"]
                return {"type": "image_url", "image_url": {
                    "url": f"data:{src['media_type']};base64,{src['data']}"
                }}
            if block["type"] == "document":
                pdf_text = block.get("_pdf_text", "[PDF content unavailable]")
                filename = block.get("_filename", "document.pdf")
                return {"type": "text", "text": f"--- FILE: {filename} ---\n{pdf_text}\n"}
            return {"type": "text", "text": ""}

        xai_content = [to_xai(b) for b in user_content]
        messages    = [{"role": "system", "content": system_prompt}]
        for h in (req.history or []):
            messages.append(h)
        messages.append({"role": "user", "content": xai_content})

        response = client.chat.completions.create(model=req.model, messages=messages, max_tokens=4096)
        return {"answer": response.choices[0].message.content}

    else:
        raise HTTPException(status_code=400, detail="Unknown provider")


# serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")

