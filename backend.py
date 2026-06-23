import os
import re
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, Ollama
from langchain_chroma import Chroma

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================

REPO_ROOT = r"C:\Users\User\Desktop\lazy"

app = FastAPI(title="Cursor-style AI Backend")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

llm = Ollama(model="qwen2-coder:30b")


# =========================
# REQUEST MODELS
# =========================

class Query(BaseModel):
    text: str


class BatchQuery(BaseModel):
    queries: List[str]


class EditBlock(BaseModel):
    file: str
    search: str
    replace: str


# =========================
# FILE EDITOR (Cursor-style)
# =========================

def parse_edit_blocks(text: str) -> List[EditBlock]:
    """
    Parses AI output and extracts edit blocks safely.
    Format expected:

    FILE: path
    SEARCH:
    old code
    REPLACE:
    new code
    """
    blocks = []
    current_block = {}
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('FILE:'):
            # Save previous block if complete
            if current_block.get('file') and current_block.get('search') and current_block.get('replace'):
                blocks.append(current_block)
            current_block = {'file': line[5:].strip()}
            
        elif line.startswith('SEARCH:'):
            search_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('REPLACE:'):
                search_lines.append(lines[i])
                i += 1
            current_block['search'] = '\n'.join(search_lines).rstrip()
            continue
            
        elif line.startswith('REPLACE:'):
            replace_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('FILE:'):
                replace_lines.append(lines[i])
                i += 1
            current_block['replace'] = '\n'.join(replace_lines).rstrip()
            continue
        
        i += 1
    
    # Don't forget the last block
    if current_block.get('file') and current_block.get('search') and current_block.get('replace'):
        blocks.append(current_block)
    
    return blocks


def apply_edit_blocks(text: str):
    """
    Applies parsed edit blocks to files with safety checks.
    Returns status for each edit.
    """
    blocks = parse_edit_blocks(text)
    results = []

    for block in blocks:
        file_path = block['file'].strip()
        old = block['search']
        new = block['replace']

        # Safety: verify real path is within repo root
        try:
            real_path = os.path.realpath(file_path)
            real_repo = os.path.realpath(REPO_ROOT)
            
            if not real_path.startswith(real_repo):
                logger.warning(f"Blocked edit outside repo: {file_path}")
                results.append({"file": file_path, "status": "blocked", "reason": "outside repo root"})
                continue
        except Exception as e:
            logger.error(f"Path validation error for {file_path}: {e}")
            results.append({"file": file_path, "status": "error", "reason": str(e)})
            continue

        # Check file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            results.append({"file": file_path, "status": "not_found"})
            continue

        # Read file
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            results.append({"file": file_path, "status": "error", "reason": f"read failed: {e}"})
            continue

        # Check if search string exists
        if old not in content:
            logger.warning(f"Search string not found in {file_path}")
            results.append({
                "file": file_path, 
                "status": "search_not_found",
                "reason": "old code not found in file"
            })
            continue

        # Apply edit
        try:
            updated = content.replace(old, new)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated)
            
            logger.info(f"Updated {file_path}")
            results.append({"file": file_path, "status": "updated"})
        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            results.append({"file": file_path, "status": "error", "reason": f"write failed: {e}"})

    return results


# =========================
# AI AGENT CORE
# =========================

def run_agent(question: str, context: str):
    """
    Runs the AI agent with the given question and context.
    Returns the raw LLM response.
    """
    prompt = f"""You are a senior software engineer working inside a live codebase.

You can read and modify files.

If you want to edit code, respond ONLY in this format:

FILE: full/path/to/file
SEARCH:
<exact existing code>
REPLACE:
<new code>

Rules:
- Make minimal changes
- Always use exact existing code in SEARCH blocks
- Do NOT explain edits inside the blocks
- Be precise with indentation and whitespace
- Only edit files that exist in the provided context

Context from repository:
{context}

User request:
{question}

Respond with your solution. If you make edits, use the FILE/SEARCH/REPLACE format above."""

    try:
        response = llm.invoke(prompt)
        logger.info("Agent response generated successfully")
        return response
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise


# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "Cursor-style AI backend running"}


@app.get("/config")
def get_config():
    """Returns backend configuration."""
    return {
        "model": "qwen2-coder:30b",
        "embedding_model": "nomic-embed-text",
        "repo_root": REPO_ROOT,
        "chunk_size": 1000,
        "status": "ready"
    }


@app.post("/agent_run")
async def agent_run(q: Query):
    """
    Main agent endpoint.
    Takes a user query, retrieves relevant code context via RAG,
    and runs the AI agent to generate a response.
    Automatically applies any edits specified in the response.
    """
    if not q.text or not q.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        logger.info(f"Received query: {q.text[:100]}...")
        
        # Retrieve relevant code from repo
        docs = db.similarity_search(q.text, k=8)
        
        if not docs:
            logger.warning("No similar documents found in vector DB")
            context = "[No matching code found in repository. Empty repository?]"
        else:
            context = "\n\n".join(
                [f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs]
            )
            logger.info(f"Retrieved {len(docs)} relevant documents")
        
        # Run AI
        response = run_agent(q.text, context)
        logger.info("Agent ran successfully")
        
        # Apply edits automatically (Cursor behavior)
        edits = apply_edit_blocks(response)
        logger.info(f"Applied {len(edits)} edits")
        
        return {
            "status": "success",
            "response": response,
            "edits": edits,
            "context_docs": len(docs)
        }
        
    except Exception as e:
        logger.error(f"Agent run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent run failed: {str(e)}")


@app.post("/agent_run/batch")
async def agent_run_batch(q: BatchQuery):
    """
    Batch processing endpoint.
    Processes multiple queries sequentially and returns all results.
    """
    if not q.queries or len(q.queries) == 0:
        raise HTTPException(status_code=400, detail="Queries list cannot be empty")
    
    results = []
    
    try:
        for idx, query in enumerate(q.queries):
            logger.info(f"Processing batch query {idx + 1}/{len(q.queries)}")
            
            # Reuse the single query logic
            result = await agent_run(Query(text=query))
            result['batch_index'] = idx
            results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} queries")
        return {
            "status": "success",
            "total": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.post("/parse_edits")
async def parse_edits(q: Query):
    """
    Endpoint to parse edits without running the full agent.
    Useful for previewing what edits would be made.
    """
    try:
        blocks = parse_edit_blocks(q.text)
        logger.info(f"Parsed {len(blocks)} edit blocks")
        
        return {
            "status": "success",
            "edits_found": len(blocks),
            "edits": [
                {
                    "file": b.get('file'),
                    "search_length": len(b.get('search', '')),
                    "replace_length": len(b.get('replace', ''))
                }
                for b in blocks
            ]
        }
        
    except Exception as e:
        logger.error(f"Edit parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Edit parsing failed: {str(e)}")


@app.get("/health")
def health_check():
    """Detailed health check."""
    try:
        # Try to access DB
        db.similarity_search("test", k=1)
        db_status = "ok"
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        db_status = "error"
    
    return {
        "status": "healthy",
        "database": db_status,
        "repo_root_exists": os.path.exists(REPO_ROOT)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
