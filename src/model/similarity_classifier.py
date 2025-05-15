from collections import Counter
from enum import Enum
from typing import List, Optional, Tuple

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances

from src.dataset.model import BongardImage


class Answer(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


DISTANCE_FNS = {
    "cosine": lambda a, b: pairwise_distances(a, b, metric="cosine"),
    "euclidean": lambda a, b: pairwise_distances(a, b, metric="euclidean"),
}
AGGREGATION_FNS = {
    "mean": lambda x: x.mean(),
    "min": lambda x: x.min(),
    "max": lambda x: x.max(),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
image_encoder, preprocess = clip.load("ViT-L/14", device=device)
image_encoder = image_encoder.eval()

text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_encoder = text_encoder.eval()


def encode_images(bongard_images: List[BongardImage]) -> np.array:
    images = [
        preprocess(Image.open(bongard_image.path)).unsqueeze(0).to(device)
        for bongard_image in bongard_images
    ]
    with torch.no_grad():
        images = torch.cat(images, dim=0)
        features = image_encoder.encode_image(images)
        return features.cpu().numpy()


def encode_texts(texts: List[str]) -> np.array:
    with torch.no_grad():
        return text_encoder.encode(texts)


def summarize(y_true: np.array, y_pred: np.array) -> str:
    num_correct = sum([y_t == y_p for y_t, y_p in zip(y_true, y_pred)])
    n = len(y_true)
    metrics = {
        "num_correct_answers": num_correct,
        "num_all_answers": n,
        "acc": round(num_correct / n * 100, 2),
    }

    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()
    pred_support = Counter(y_pred)
    df_report["predicted_support"] = df_report.index.map(
        lambda label: (
            pred_support[label] if "LEFT" in label or "RIGHT" in label else 0
        )
    )

    summary = f"{metrics}\n{df_report.to_string(float_format='%.2f')}"
    return summary.strip()


def classify_to_side(
    left: np.ndarray,
    right: np.ndarray,
    test: np.ndarray,
    distance_fn: str,
    aggregation_fn: str,
    top_k: Optional[int] = None,
) -> Answer:
    if top_k is None:
        top_k = max(len(left), len(right))
    dis_fn = DISTANCE_FNS[distance_fn]
    agg_fn = AGGREGATION_FNS[aggregation_fn]
    tl = dis_fn(test, left)
    tr = dis_fn(test, right)
    # Get top k values
    tl, tr = [agg_fn(np.sort(x[0])[:top_k]) for x in [tl, tr]]
    return Answer.LEFT if tl < tr else Answer.RIGHT


def classify_to_sides(
    left: np.ndarray,
    right: np.ndarray,
    distance_fn: str,
    aggregation_fn: str,
    top_k: Optional[int] = None,
) -> Tuple[Answer, Answer]:
    if top_k is None:
        top_k = max(len(left), len(right)) - 1
    dis_fn = DISTANCE_FNS[distance_fn]
    agg_fn = AGGREGATION_FNS[aggregation_fn]
    ll = dis_fn(left[-1:], left[:-1])
    lr = dis_fn(left[-1:], right[:-1])
    rl = dis_fn(right[-1:], left[:-1])
    rr = dis_fn(right[-1:], right[:-1])
    # Get top k values
    ll, lr, rl, rr = [np.sort(x[0])[:top_k] for x in [ll, lr, rl, rr]]
    lr_score = agg_fn(ll) + agg_fn(rr)
    rl_score = agg_fn(lr) + agg_fn(rl)
    return (
        (Answer.LEFT, Answer.RIGHT)
        if lr_score < rl_score
        else (Answer.RIGHT, Answer.LEFT)
    )
