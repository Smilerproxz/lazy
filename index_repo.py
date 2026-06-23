import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================

REPO_PATH = r"C:\Users\User\Desktop\lazy"  # your code folder

IGNORE = {"node_modules", ".git", "__pycache__", "dist", "build"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

# =========================
# HELPER FUNCTIONS
# =========================

def read_file(path):
    """Safely read file with error handling."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None


def index_repo():
    """
    Indexes all code files in the repository.
    Splits files into chunks and stores in Chroma vector DB.
    """
    docs = []
    metas = []
    file_count = 0
    chunk_count = 0
    error_count = 0

    # Walk through repository
    for root, dirs, files in os.walk(REPO_PATH):

        # Skip junk folders
        dirs[:] = [d for d in dirs if d not in IGNORE]

        for file in files:
            # Only process code files
            if file.endswith((".py", ".js", ".ts", ".tsx", ".html", ".css", ".json", ".md")):

                path = os.path.join(root, file)
                
                try:
                    text = read_file(path)
                    
                    if text is None or len(text.strip()) == 0:
                        logger.warning(f"Skipped empty file: {path}")
                        continue
                    
                    # Split into chunks
                    chunks = splitter.split_text(text)
                    
                    if not chunks:
                        logger.warning(f"No chunks generated for: {path}")
                        continue
                    
                    # Add to docs
                    for chunk in chunks:
                        docs.append(chunk)
                        metas.append({"source": path})
                        chunk_count += 1
                    
                    file_count += 1
                    logger.info(f"Indexed {path} ({len(chunks)} chunks)")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing {path}: {e}")
                    continue

    # Add all docs to database
    if docs:
        try:
            db.add_texts(docs, metadatas=metas)
            logger.info(f"✅ Repo indexed successfully")
            logger.info(f"   Files indexed: {file_count}")
            logger.info(f"   Chunks created: {chunk_count}")
            logger.info(f"   Errors: {error_count}")
        except Exception as e:
            logger.error(f"Failed to add texts to database: {e}")
            return False
    else:
        logger.warning("No documents to index")
        return False

    return True


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    logger.info(f"Starting repository indexing from: {REPO_PATH}")
    
    # Check if repo path exists
    if not os.path.exists(REPO_PATH):
        logger.error(f"Repository path does not exist: {REPO_PATH}")
        exit(1)
    
    # Index the repo
    success = index_repo()
    
    if success:
        logger.info("✅ Indexing complete")
    else:
        logger.error("❌ Indexing failed")
        exit(1)
