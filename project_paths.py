from pathlib import Path


ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "epstein files"
RESULTS_JSON = ROOT / "results.json"
FINAL_DATA_JSON = ROOT / "final_data.json"
CHUNKS_JSON = ROOT / "chunks.json"
FAISS_DIR = ROOT / "faiss_index"
FAISS_INDEX = FAISS_DIR / "index.faiss"
FAISS_CHUNKS = FAISS_DIR / "chunks.pkl"
IMAGES_DIR = ROOT / "images"

