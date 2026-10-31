"""Build a model card, a nutrition label and a json record from any fitted
sklearn classifier. Sections follow Mitchell et al. (2019) and the technical
documentation fields in Annex IV of the EU AI Act.
"""

import json
import os
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


def subgroup_metrics(y_true, y_pred, y_prob, groups):
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        row = {
            "group": str(g),
            "n": int(m.sum()),
            "accuracy": accuracy_score(yt, yp),
            "selection_rate": float(yp.mean()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
        }
        if y_prob is not None and len(np.unique(yt)) > 1:
            row["auc"] = roc_auc_score(yt, y_prob[m])
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        row["fpr"] = fp / (fp + tn) if fp + tn else 0.0
        row["fnr"] = fn / (fn + tp) if fn + tp else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_summary(sub):
    """Common ratios and gaps between the best and worst treated group."""
    rates = sub.set_index("group")["selection_rate"]
    out = {
        "disparate_impact_ratio": float(rates.min() / rates.max()) if rates.max() > 0 else None,
        "demographic_parity_gap": float(rates.max() - rates.min()),
        "equal_opportunity_gap": float(sub["recall"].max() - sub["recall"].min()),
        "fpr_gap": float(sub["fpr"].max() - sub["fpr"].min()),
    }
    out["four_fifths_rule_pass"] = out["disparate_impact_ratio"] is not None and out["disparate_impact_ratio"] >= 0.8
    return out


def describe_data(X, y, sensitive):
    desc = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(np.mean(y)),
        "feature_names": list(map(str, X.columns)) if hasattr(X, "columns") else [str(i) for i in range(X.shape[1])],
        "sensitive_attribute": sensitive.name if hasattr(sensitive, "name") else "sensitive",
        "sensitive_groups": {str(k): int(v) for k, v in pd.Series(sensitive).value_counts().items()},
    }
    return desc


def build_card(model, X, y, sensitive, meta, threshold=0.5):
    """Return a dict with everything the markdown and json writers need.

    model: fitted sklearn classifier with predict_proba
    X, y: evaluation data (ideally a held out test set)
    sensitive: a Series aligned with X giving the protected group
    meta: dict with name, version, owner, intended_use, out_of_scope, data_source,
          training_period, limitations, contact
    """
    y = np.asarray(y)
    sensitive = pd.Series(np.asarray(sensitive), name=getattr(sensitive, "name", "sensitive"))
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = None
    pred = (prob >= threshold).astype(int) if prob is not None else model.predict(X)

    overall = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": threshold,
    }
    if prob is not None:
        overall["auc"] = roc_auc_score(y, prob)

    sub = subgroup_metrics(y, pred, prob, sensitive.values)
    fair = fairness_summary(sub)

    card = {
        "model_details": {
            "name": meta.get("name", type(model).__name__),
            "version": meta.get("version", "0.1"),
            "owner": meta.get("owner", ""),
            "contact": meta.get("contact", ""),
            "model_type": type(model).__name__,
            "parameters": {k: str(v) for k, v in model.get_params().items()},
            "generated_on": datetime.date.today().strftime("%d %B %Y"),
        },
        "intended_use": meta.get("intended_use", ""),
        "out_of_scope_use": meta.get("out_of_scope", ""),
        "training_data": dict(describe_data(X, y, sensitive),
                              source=meta.get("data_source", ""),
                              period=meta.get("training_period", "")),
        "performance_overall": {k: round(float(v), 4) for k, v in overall.items()},
        "performance_by_group": sub.round(4).to_dict(orient="records"),
        "fairness": fair,
        "limitations": meta.get("limitations", ""),
        "human_oversight": meta.get("human_oversight", ""),
        "monitoring": meta.get("monitoring", ""),
    }
    return card


def nutrition_grade(card):
    """Letter grade the way a food label would, from a few blunt checks."""
    f = card["fairness"]
    p = card["performance_overall"]
    points = 0
    if f["four_fifths_rule_pass"]:
        points += 1
    if f["equal_opportunity_gap"] < 0.1:
        points += 1
    if p.get("auc", 0) >= 0.75:
        points += 1
    if card["limitations"]:
        points += 1
    if card["human_oversight"]:
        points += 1
    return {5: "A", 4: "B", 3: "C", 2: "D"}.get(points, "E")


def write_markdown(card, path):
    d = card["model_details"]
    t = card["training_data"]
    p = card["performance_overall"]
    f = card["fairness"]
    L = []
    L.append("# Model card: %s (v%s)" % (d["name"], d["version"]))
    L.append("")
    L.append("Generated %s. Owner: %s. Contact: %s. Model type: %s."
             % (d["generated_on"], d["owner"], d["contact"], d["model_type"]))
    L.append("")
    L.append("## Intended use")
    L.append("")
    L.append(card["intended_use"])
    L.append("")
    L.append("## Out of scope use")
    L.append("")
    L.append(card["out_of_scope_use"])
    L.append("")
    L.append("## Training and evaluation data")
    L.append("")
    L.append("Source: %s. Period: %s. Evaluation rows: %d. Features: %d. Positive rate: %.3f."
             % (t["source"], t["period"], t["n_rows"], t["n_features"], t["positive_rate"]))
    L.append("")
    L.append("Sensitive attribute used for subgroup analysis: %s. Group counts: %s."
             % (t["sensitive_attribute"], ", ".join("%s=%d" % kv for kv in t["sensitive_groups"].items())))
    L.append("")
    L.append("Features: " + ", ".join(t["feature_names"]))
    L.append("")
    L.append("## Performance (overall)")
    L.append("")
    for k, v in p.items():
        L.append("%s: %s" % (k, v))
    L.append("")
    L.append("## Performance by group")
    L.append("")
    cols = list(card["performance_by_group"][0].keys())
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for row in card["performance_by_group"]:
        L.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    L.append("")
    L.append("## Fairness")
    L.append("")
    L.append("Disparate impact ratio (min selection rate over max): %.3f. Four fifths rule: %s."
             % (f["disparate_impact_ratio"], "pass" if f["four_fifths_rule_pass"] else "fail"))
    L.append("Demographic parity gap: %.3f. Equal opportunity gap (recall): %.3f. False positive rate gap: %.3f."
             % (f["demographic_parity_gap"], f["equal_opportunity_gap"], f["fpr_gap"]))
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append(card["limitations"])
    L.append("")
    L.append("## Human oversight")
    L.append("")
    L.append(card["human_oversight"])
    L.append("")
    L.append("## Monitoring")
    L.append("")
    L.append(card["monitoring"])
    L.append("")
    L.append("## Annex IV mapping")
    L.append("")
    L.append("1. General description: see Intended use and Model details.")
    L.append("2. Development process and design choices: see Model details (parameters) and Training data.")
    L.append("3. Monitoring, functioning and control: see Human oversight and Monitoring.")
    L.append("4. Performance metrics and appropriateness: see Performance sections.")
    L.append("5. Risk management: see Fairness and Limitations.")
    L.append("6. Changes through lifecycle: version field, regenerate this card on each retrain.")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def write_label(card, path):
    """Short label in the style of a food nutrition panel."""
    d = card["model_details"]
    p = card["performance_overall"]
    f = card["fairness"]
    t = card["training_data"]
    grade = nutrition_grade(card)
    L = []
    L.append("# AI nutrition label")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append("| Model | %s v%s |" % (d["name"], d["version"]))
    L.append("| Type | %s |" % d["model_type"])
    L.append("| Owner | %s |" % d["owner"])
    L.append("| Made for | %s |" % card["intended_use"].split(".")[0])
    L.append("| Not for | %s |" % card["out_of_scope_use"].split(".")[0])
    L.append("| Data | %s, %d rows, %d features |" % (t["source"], t["n_rows"], t["n_features"]))
    L.append("| Accuracy | %.1f%% |" % (100 * p["accuracy"]))
    if "auc" in p:
        L.append("| AUC | %.3f |" % p["auc"])
    L.append("| Disparate impact ratio | %.2f (%s) |" % (f["disparate_impact_ratio"], "pass" if f["four_fifths_rule_pass"] else "fail"))
    L.append("| Equal opportunity gap | %.3f |" % f["equal_opportunity_gap"])
    L.append("| Human in the loop | %s |" % ("yes" if card["human_oversight"] else "no"))
    L.append("| Known limitations | %s |" % ("documented" if card["limitations"] else "none listed"))
    L.append("| Overall grade | %s |" % grade)
    L.append("")
    L.append("Grade is out of five checks: four fifths rule, equal opportunity gap under 0.10, AUC at least 0.75, limitations documented, human oversight documented. A means all five pass.")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def write_all(card, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    write_markdown(card, os.path.join(out_dir, "model_card.md"))
    write_label(card, os.path.join(out_dir, "nutrition_label.md"))
    with open(os.path.join(out_dir, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2, default=str)
