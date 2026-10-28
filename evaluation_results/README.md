# Evaluation results

Results cover the same 59 annotated articles for 13 model/strategy combinations, plus four EMI-only evaluations of multi-property outputs. Each of the 17 configurations includes:

- `summary_table.json`: aggregate counts and precision, recall, and F1.
- `per_article.json`: scores for every article.
- `details.json`: sample assignments and scoring details.

Single-property results use EMI shielding effectiveness; other strategies use all 13 targets. Each multi-property directory also contains an `emi/` comparison. Missing or empty predictions remain in the evaluation. The annotation reader preserves repeated measurements without changing the original files.

## Reproduce the results

After installing the project, run from the repository root:

```bash
python evaluation_results/reproduce.py
```

This verifies all 51 result files using the current evaluator, released annotations and outputs, and recorded matching decisions. It checks file hashes, requires no API key, and makes no network calls. To save a separate copy:

```bash
python evaluation_results/reproduce.py --output reproduced_evaluation
```

Tested with Python 3.12.6, NumPy 2.4.4, and SciPy 1.17.1.

Saved [output settings and initialization records](replay/provenance.json), [material-decision sources](replay/material_provenance.json), and [property mappings](replay/property_provenance.json) document the inputs to this evaluation. Reproduction uses these fixed decisions rather than new judge calls.
