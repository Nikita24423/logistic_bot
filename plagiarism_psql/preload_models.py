"""Pre-download HuggingFace models at image build time (cached Docker layer)."""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ENCODER = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
LM = os.getenv("AI_MODEL", "sberbank-ai/rugpt3small_based_on_gpt2")

print(f"Downloading encoder: {ENCODER}")
SentenceTransformer(ENCODER, device="cpu")

print(f"Downloading LM: {LM}")
AutoTokenizer.from_pretrained(LM)
AutoModelForCausalLM.from_pretrained(LM)

print("Model preload complete.")
