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
import google.generativeai as genai
import fitz  # pymupdf – PDF text extraction for models that can't read raw PDFs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_ROOT = os.environ.get("DOCS_ROOT", "/docs")

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
    model:    str  # "claude" | "gpt4o" | "gemini"
    history:  Optional[list] = []


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/projects")
def get_projects():
    return {"projects": list_projects()}


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
    if req.model == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
        client   = anthropic.Anthropic(api_key=api_key)
        # Strip internal keys from blocks before sending to Anthropic
        clean_content = []
        for block in user_content:
            b = {k: v for k, v in block.items() if not k.startswith("_")}
            clean_content.append(b)
        messages = (req.history or []) + [{"role": "user", "content": clean_content}]
        response = client.messages.create(
            model      = "claude-sonnet-4-5",
            max_tokens = 4096,
            system     = system_prompt,
            messages   = messages,
        )
        return {"answer": response.content[0].text}

    # ── GPT-4o ───────────────────────────────────────────────────────────────
    elif req.model == "gpt4o":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        client = openai.OpenAI(api_key=api_key)

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

        response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=4096)
        return {"answer": response.choices[0].message.content}

    # ── Gemini ────────────────────────────────────────────────────────────────
    elif req.model == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name    = "gemini-1.5-pro",
            system_instruction = system_prompt,
        )

        # Gemini parts for the current message
        parts = []
        for block in user_content:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "image":
                src = block["source"]
                parts.append({"mime_type": src["media_type"],
                               "data": base64.b64decode(src["data"])})
            elif block["type"] == "document":
                parts.append({"mime_type": "application/pdf",
                               "data": base64.b64decode(block["source"]["data"])})

        # Build conversation history for Gemini
        gemini_history = []
        for h in (req.history or []):
            role = "user" if h["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [h["content"]]})

        if gemini_history:
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(parts)
        else:
            response = model.generate_content(parts)

        return {"answer": response.text}

    else:
        raise HTTPException(status_code=400, detail="Unknown model")


# serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
