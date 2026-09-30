from sentence_transformers import CrossEncoder
import torch

cross_encoder_model=CrossEncoder(
    "Qwen/Qwen3-Reranker-0.6B",
    device="cuda" if torch.cuda.is_available() else "cpu"
)