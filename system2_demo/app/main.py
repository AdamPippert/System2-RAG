from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.engine import answer_question
from app.llm.providers import LLMClient
from app.rag.index import SimpleIndex
from app.rag.text import read_pdf, read_text, chunk_text


app = FastAPI(title="System2 Demo")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def get_index() -> SimpleIndex:
    return SimpleIndex(settings.data_dir)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index = get_index()
    return templates.TemplateResponse("index.html", {"request": request, "index_size": index.size})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    index = get_index()
    data_dir = Path(settings.data_dir)
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    path = uploads / file.filename
    content = await file.read()
    path.write_bytes(content)

    ext = path.suffix.lower()
    if ext == ".pdf":
        text = read_pdf(str(path))
    else:
        text = content.decode("utf-8", errors="ignore")

    chunks = chunk_text(text, source_id=file.filename, chunk_chars=settings.chunk_chars, overlap=settings.chunk_overlap)
    index.add_chunks(chunks)

    return templates.TemplateResponse(
        "partials/upload_result.html",
        {"request": request, "filename": file.filename, "added": len(chunks), "index_size": index.size},
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    index = get_index()
    llm = LLMClient()

    pkg = await answer_question(index, llm, question=question)

    return templates.TemplateResponse(
        "partials/answer.html",
        {
            "request": request,
            "question": question,
            "pkg": pkg,
        },
    )
