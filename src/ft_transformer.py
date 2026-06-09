"""FT-Transformer comparison model (Gorishniy et al., 2021).

A modern tabular deep-learning baseline to contrast with gradient boosting.
FT-Transformer tokenises every numeric and categorical feature into an
embedding and applies a standard Transformer encoder. The honest, current
(2024-2026) finding is that GBTs still match or beat tabular transformers on
most credit datasets unless data volume is very large (Booking.com, arXiv
2405.13692) — this module exists to *demonstrate* that comparison, not to win.

Kept deliberately small/optional: it trains on the numeric feature block with
simple median-impute + standardisation and a short schedule, so it runs on the
RTX 3060 without dominating the project's runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prep_numeric(X: pd.DataFrame, num_cols: list[str], medians=None, stds=None,
                  means=None):
    Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = Xn.median()
        means = Xn.mean()
        stds = Xn.std().replace(0, 1.0)
    Xn = Xn.fillna(medians)
    Xn = (Xn - means) / stds
    return Xn.to_numpy(dtype=np.float32), medians, stds, means


def train_ft_transformer(X_tr, y_tr, X_va, y_va, num_cols, epochs: int = 8,
                         batch_size: int = 1024, lr: float = 1e-3):
    """Train a compact FT-Transformer; returns ``(predict_fn, history)``.

    ``predict_fn(X_df) -> np.ndarray`` yields probabilities of default.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from rtdl_revisiting_models import FTTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, med, std, mean = _prep_numeric(X_tr, num_cols)
    Xva, *_ = _prep_numeric(X_va, num_cols, med, std, mean)

    ds = TensorDataset(torch.from_numpy(Xtr),
                       torch.from_numpy(y_tr.to_numpy(dtype=np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = FTTransformer(
        n_cont_features=len(num_cols), cat_cardinalities=[], d_out=1,
        n_blocks=2, d_block=128, attention_n_heads=8,
        attention_dropout=0.2, ffn_d_hidden_multiplier=4 / 3, ffn_dropout=0.1,
        residual_dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb, None).squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
        history.append(float(loss.item()))

    def predict_fn(X_df: pd.DataFrame) -> np.ndarray:
        Xq, *_ = _prep_numeric(X_df, num_cols, med, std, mean)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xq).to(device), None).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    return predict_fn, history
