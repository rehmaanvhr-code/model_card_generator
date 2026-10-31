import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_card
import generate


def fitted_example():
    X, y, g = generate.make_credit_data(n=800, seed=3)
    model = LogisticRegression(max_iter=500).fit(X, y)
    meta = {"name": "t", "version": "0.1", "intended_use": "test. more",
            "out_of_scope": "none. more", "limitations": "some", "human_oversight": "yes"}
    return model, X, y, g, meta


def test_card_has_required_sections():
    model, X, y, g, meta = fitted_example()
    card = model_card.build_card(model, X, y, g, meta)
    for key in ["model_details", "intended_use", "out_of_scope_use", "training_data",
                "performance_overall", "performance_by_group", "fairness", "limitations"]:
        assert key in card
    assert len(card["performance_by_group"]) == 2


def test_disparate_impact_between_zero_and_one():
    model, X, y, g, meta = fitted_example()
    card = model_card.build_card(model, X, y, g, meta)
    di = card["fairness"]["disparate_impact_ratio"]
    assert 0 <= di <= 1


def test_writers_produce_files(tmp_path):
    model, X, y, g, meta = fitted_example()
    card = model_card.build_card(model, X, y, g, meta)
    model_card.write_all(card, str(tmp_path))
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "nutrition_label.md").exists()
    loaded = json.load(open(tmp_path / "model_card.json"))
    assert loaded["model_details"]["name"] == "t"


def test_grade_is_a_letter():
    model, X, y, g, meta = fitted_example()
    card = model_card.build_card(model, X, y, g, meta)
    assert model_card.nutrition_grade(card) in list("ABCDE")
