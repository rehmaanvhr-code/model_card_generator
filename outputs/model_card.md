# Model card: consumer_credit_default (v1.0)

Generated 31 October 2026. Owner: Credit risk analytics. Contact: model.owner@example.com. Model type: GradientBoostingClassifier.

## Intended use

Score new unsecured consumer loan applications for probability of default within 12 months. The score supports an underwriter decision and is not the sole basis for decline.

## Out of scope use

Not for pricing, collections prioritisation, marketing, or any decision about existing customers. Not validated for applicants outside the 21 to 69 age range or for business loans.

## Training and evaluation data

Source: Synthetic applications generated in generate.py with a fixed seed. Period: n/a (synthetic). Evaluation rows: 1800. Features: 6. Positive rate: 0.155.

Sensitive attribute used for subgroup analysis: group. Group counts: A=1179, B=621.

Features: income_k, age, debt_ratio, history_years, late_payments, loan_amount_k

## Performance (overall)

accuracy: 0.84
precision: 0.485
recall: 0.5233
f1: 0.5034
threshold: 0.3
auc: 0.8225

## Performance by group

| group | n | accuracy | selection_rate | precision | recall | auc | fpr | fnr |
|---|---|---|---|---|---|---|---|---|
| A | 1179 | 0.8584 | 0.1357 | 0.4438 | 0.4765 | 0.8232 | 0.0864 | 0.5235 |
| B | 621 | 0.8052 | 0.2271 | 0.5319 | 0.5769 | 0.8058 | 0.1344 | 0.4231 |

## Fairness

Disparate impact ratio (min selection rate over max): 0.598. Four fifths rule: fail.
Demographic parity gap: 0.091. Equal opportunity gap (recall): 0.100. False positive rate gap: 0.048.

## Limitations

Group B has lower average income in the data, so the model declines group B applicants at a higher rate. The protected attribute is not a model input but is correlated with income. No external validation set. Scores drift if the income distribution of applicants shifts.

## Human oversight

Underwriter reviews every decline with score between 0.4 and 0.6 and can override with a documented reason. Overrides are logged and reviewed monthly.

## Monitoring

Monthly report of AUC, selection rate by group and disparate impact ratio. Escalate to model risk if AUC falls below 0.70 or the disparate impact ratio falls below 0.80.

## Annex IV mapping

1. General description: see Intended use and Model details.
2. Development process and design choices: see Model details (parameters) and Training data.
3. Monitoring, functioning and control: see Human oversight and Monitoring.
4. Performance metrics and appropriateness: see Performance sections.
5. Risk management: see Fairness and Limitations.
6. Changes through lifecycle: version field, regenerate this card on each retrain.
