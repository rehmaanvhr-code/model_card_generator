# Model card generator

A small library that takes any fitted scikit-learn classifier, an evaluation set and a protected attribute, and writes three things: a full model card in markdown, a one page AI nutrition label, and a json record that can be loaded into a model inventory. The card sections follow the model card fields proposed by Mitchell et al. (2019) and are mapped at the end to the technical documentation items in Annex IV of the EU AI Act. The demo script trains a gradient boosted credit default model on synthetic loan data and documents it.

## Connection to the research

The nutrition label format comes from the AI Nutrition Label I built for MMAI 803 AI Ethics and Policy, where the idea was that a decision model should carry a short, standard panel of facts the way packaged food does: what it is for, what it is not for, what went into it, how well it performs, and whether it treats groups differently. This repo turns that paper exercise into code that regenerates the label every time the model is retrained, so the documentation cannot drift away from the model. It also supports the argument in "The Governance Lag: Why Slow AI Adoption Is the Larger Risk" (Rehman, 2026) that most governance cost is avoidable if documentation is produced by the pipeline rather than written by hand after the fact.

## Why this matters in financial services

Documentation is the first control any model risk framework enforces. OSFI Guideline E-23 expects documentation that lets an independent reviewer understand the model, the EU AI Act makes credit scoring a high risk use that needs Annex IV technical documentation, and the Colorado AI Act requires developers to tell deployers about intended use, data, performance and discrimination risk. A generator like this one produces a consistent first draft for every model in an inventory and makes subgroup performance and the four fifths rule a default part of the record rather than an afterthought.

## How to run

```
pip install -r requirements.txt
python generate.py
pytest
```

To document your own model:

```python
import model_card
card = model_card.build_card(model, X_test, y_test, sensitive_series, meta, threshold=0.5)
model_card.write_all(card, "outputs")
```

`meta` is a plain dict with name, version, owner, contact, intended_use, out_of_scope, data_source, training_period, limitations, human_oversight and monitoring. The model needs `predict_proba`.

## Outputs

`outputs/model_card.md` has model details, intended and out of scope use, data summary, overall performance, a performance table by group (accuracy, selection rate, precision, recall, AUC, false positive and false negative rates), fairness metrics, limitations, oversight, monitoring and an Annex IV mapping. `outputs/nutrition_label.md` is the short panel with a letter grade from five blunt checks. `outputs/model_card.json` is the same content as data. `outputs/group_performance.png` plots selection rate, recall and false positive rate by group.

In the demo the synthetic data gives group B a lower average income, and the model flags group B applicants as likely defaulters at a higher rate even though group is not a model input. The disparate impact ratio comes out around 0.6 and fails the four fifths rule, which is the point of the demo: the card surfaces the problem without anyone having to go looking for it.

## Data

The demo data is generated in `generate.py` with a fixed seed. No real applicants. The data generating process is written out in the code so the unfairness is traceable to the income gap between groups.

## Limitations

Binary classification only. The fairness metrics are group level rates and say nothing about individual fairness or about proxies beyond the one attribute passed in. The letter grade is a teaching device and the thresholds in it are not regulatory thresholds, apart from the four fifths rule which is a US employment selection guideline that has been borrowed widely.

Saif Rehman, MMAI, Smith School of Business, Queen's University
