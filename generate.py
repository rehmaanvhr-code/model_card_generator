"""Train a small credit default model on synthetic data and write a model card,
a nutrition label and a json record for it. Run from the repo root:

    python generate.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model_card

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def make_credit_data(n=6000, seed=7):
    """Synthetic loan applications with a protected attribute that is
    correlated with income so the model has a chance to be unfair."""
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.65, 0.35])
    income = np.where(group == "A", rng.normal(72, 20, n), rng.normal(58, 18, n)).clip(15, 250)
    age = rng.integers(21, 70, n)
    debt_ratio = rng.beta(2, 5, n)
    history_years = rng.integers(0, 30, n)
    late_payments = rng.poisson(0.8, n)
    loan_amount = rng.normal(25, 10, n).clip(2, 80)
    logit = (-1.0 - 0.05 * income + 5.0 * debt_ratio - 0.06 * history_years
             + 0.9 * late_payments + 0.03 * loan_amount)
    p = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p)
    X = pd.DataFrame({
        "income_k": income.round(1),
        "age": age,
        "debt_ratio": debt_ratio.round(3),
        "history_years": history_years,
        "late_payments": late_payments,
        "loan_amount_k": loan_amount.round(1),
    })
    return X, pd.Series(default, name="default"), pd.Series(group, name="group")


def plot_groups(card, path):
    rows = card["performance_by_group"]
    groups = [r["group"] for r in rows]
    metrics = ["selection_rate", "recall", "fpr"]
    x = np.arange(len(groups))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, [r[m] for r in rows], w, label=m)
    ax.set_xticks(x + w)
    ax.set_xticklabels(groups)
    ax.set_ylabel("rate")
    ax.set_xlabel("group")
    ax.set_title("Performance by protected group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y, group = make_credit_data()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, group, test_size=0.3, random_state=1, stratify=y)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=1)
    model.fit(X_tr, y_tr)

    meta = {
        "name": "consumer_credit_default",
        "version": "1.0",
        "owner": "Credit risk analytics",
        "contact": "model.owner@example.com",
        "intended_use": "Score new unsecured consumer loan applications for probability of default within 12 months. The score supports an underwriter decision and is not the sole basis for decline.",
        "out_of_scope": "Not for pricing, collections prioritisation, marketing, or any decision about existing customers. Not validated for applicants outside the 21 to 69 age range or for business loans.",
        "data_source": "Synthetic applications generated in generate.py with a fixed seed",
        "training_period": "n/a (synthetic)",
        "limitations": "Group B has lower average income in the data, so the model declines group B applicants at a higher rate. The protected attribute is not a model input but is correlated with income. No external validation set. Scores drift if the income distribution of applicants shifts.",
        "human_oversight": "Underwriter reviews every decline with score between 0.4 and 0.6 and can override with a documented reason. Overrides are logged and reviewed monthly.",
        "monitoring": "Monthly report of AUC, selection rate by group and disparate impact ratio. Escalate to model risk if AUC falls below 0.70 or the disparate impact ratio falls below 0.80.",
    }

    card = model_card.build_card(model, X_te, y_te, g_te, meta, threshold=0.3)
    model_card.write_all(card, OUT)
    plot_groups(card, os.path.join(OUT, "group_performance.png"))

    print("overall:", card["performance_overall"])
    print("fairness:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in card["fairness"].items()})
    print("nutrition grade:", model_card.nutrition_grade(card))
    print("wrote outputs/model_card.md, nutrition_label.md, model_card.json, group_performance.png")


if __name__ == "__main__":
    main()
