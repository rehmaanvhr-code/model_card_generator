# AI nutrition label

| Field | Value |
|---|---|
| Model | consumer_credit_default v1.0 |
| Type | GradientBoostingClassifier |
| Owner | Credit risk analytics |
| Made for | Score new unsecured consumer loan applications for probability of default within 12 months |
| Not for | Not for pricing, collections prioritisation, marketing, or any decision about existing customers |
| Data | Synthetic applications generated in generate.py with a fixed seed, 1800 rows, 6 features |
| Accuracy | 84.0% |
| AUC | 0.823 |
| Disparate impact ratio | 0.60 (fail) |
| Equal opportunity gap | 0.100 |
| Human in the loop | yes |
| Known limitations | documented |
| Overall grade | C |

Grade is out of five checks: four fifths rule, equal opportunity gap under 0.10, AUC at least 0.75, limitations documented, human oversight documented. A means all five pass.