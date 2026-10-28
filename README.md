# PolyCompIE

**Structured polymer-composite data from scientific articles.**

PolyCompIE extracts sample compositions and measured properties from article text and tables. It provides four extraction strategies, a composition-aware evaluator, and a web interface for inspecting and exporting results.

**PDF → Markdown → structured samples → evaluation**

[Installation](#installation) · [Usage guide](docs/usage.md) ·
[Prompts](src/polycompie/extraction/prompts/prompts.yaml) ·
[Deployment](DEPLOY.md)

## Extraction strategies

| Phase | Strategy | Workflow |
|---|---|---|
| 1 | Single-property extraction | One pass targeting EMI shielding effectiveness |
| 2 | Multi-property extraction | One pass targeting 13 electrical, mechanical, thermal, and fire-related properties |
| 3 | Two-step extraction | Formulation inventory, followed by property attachment |
| 4 | Chain-of-verification | Extraction, optional self-review, and per-sample evidence grounding, verification, and conditional repair |

The full CoV strategy accepts HIGH-confidence samples after self-review and verifies LOW-confidence samples. The [variant configurations](src/polycompie/extraction/prompts/variants.yaml) define the ablations and iteration limits.

Each output sample contains `sample_id`, `matrix_name`, `fillers`,
`sample_thickness`, and `properties`. Property entries include values, units, and measurement conditions. See the [JSON schema](src/polycompie/extraction/schemas/extraction_schema.json).

## Installation

Requires Python 3.10 or newer. PDF conversion runs locally; Docling may download
model weights on first use. Hosted LLM extraction requires your own API key.

```bash
git clone https://github.com/qy51oliver/polycompie.git
cd polycompie
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app]"
```

Use `python -m pip install -e .` for the CLI without the web interface.

## Quick start

```bash
python -m app.main
```

Open **http://127.0.0.1:8080**, upload a PDF, select properties, and choose single-pass or CoV extraction. The web interface supports GPT-5.4 with your OpenAI key, or local Ollama models for single-pass extraction. Results can be inspected and downloaded as JSON or Excel.

For batch processing, follow the [usage guide](docs/usage.md) to initialize the article ledger, convert PDFs, and clean the Markdown before extraction:

```bash
polycompie-extract --batch my_collection --phase 2 --model gpt-5.4
```

The batch CLI supports GPT-5.2, GPT-5.4, Claude Sonnet 4.6, and Gemini 2.5 Pro. Exact model identifiers and generation settings are in [`prompts.yaml`](src/polycompie/extraction/prompts/prompts.yaml). Docling is the default PDF parser; LiteParse is also available.

## Evaluation

```bash
polycompie-evaluate --annotated-dir data/annotated_wo_figs --experiments-dir data/experiments --phase 2 --model gpt-5.4 --property-set selected --out-dir data/eval/gpt-5.4-phase2 --llm-model gpt-4o
```

The evaluator aligns samples using Hungarian assignment with a similarity of `0.4 × matrix + 0.4 × filler + 0.2 × thickness`. It reports composition and property precision, recall, and F1 at relaxed (≥ 0.8) and strict (1.0) thresholds. Property matches require a qualifying composition match; unmatched samples contribute property errors. Measurement conditions are retained but not directly scored.

Every valid reference article is included, including articles with missing or empty predictions. See the [usage guide](docs/usage.md#evaluation) for filtering and failure handling.

The general evaluation CLI uses GPT-4o by default for name matching and requires `OPENAI_API_KEY`; `--no-llm` uses deterministic matching and can produce different scores. To reproduce the saved benchmark results with their recorded matching decisions and no API calls, follow the [reproduction instructions](evaluation_results/README.md).

## Research materials

This repository accompanies *From LLM-Based Literature Data Curation to Machine Learning for Polymer Composite Design*. Downstream machine-learning code is maintained in [PolyComp-EMISE-ML](https://github.com/gourav-k/PolyComp-EMISE-ML).

The 59-paper benchmark includes [human annotations and references](annotations/REFERENCES.md), [model outputs by extraction strategy](outputs/), and [evaluation results](evaluation_results/). The [broader EMI dataset](emi_dataset/REFERENCES.md) contains GPT-5.4 multi-property outputs and references for 247 additional papers.

## Development

```bash
python -m pip install -e ".[app,dev]"
python -m pytest tests/ -q
```

The tests use local fixtures and mocked provider calls; they do not run LLM extraction experiments. Library code is under `src/polycompie/`, the web interface under `app/`, and tests under `tests/`.

## License

Licensed under the [MIT License](LICENSE). Third-party dependencies retain their own licenses.
