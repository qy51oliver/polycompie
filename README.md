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

The full CoV strategy accepts HIGH-confidence samples after self-review and verifies LOW-confidence samples. Verification may accept a sample, request additional evidence, or trigger repair.
The [variant configurations](src/polycompie/extraction/prompts/variants.yaml)
define the ablations and iteration limits.

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

The batch CLI supports GPT-5.2, GPT-5.4, Claude Sonnet 4.6, and Gemini 2.5 Pro. Exact model identifiers and generation settings are in[`prompts.yaml`](src/polycompie/extraction/prompts/prompts.yaml). Docling is the default PDF parser; LiteParse is also available.

## Evaluation

```bash
polycompie-evaluate --annotated-dir data/annotated_wo_figs --experiments-dir data/experiments --phase 2 --model gpt-5.4 --property-set selected --out-dir data/eval/gpt-5.4-phase2 --llm-model gpt-4o
```

The evaluator aligns samples using Hungarian assignment with a similarity of `0.4 × matrix + 0.4 × filler + 0.2 × thickness`. It reports composition and property precision, recall, and F1 at relaxed (≥ 0.8) and strict (1.0) thresholds. Property matches are credited only in pairs meeting the selected threshold; unmatched samples contribute property errors. Experimental conditions are retained in the extraction but are not directly scored.

Evaluation uses GPT-4o by default for material-name matching and unresolved predicted property keys. It requires `OPENAI_API_KEY` and can incur API charges. `--no-llm` disables those calls and uses deterministic matching, which can produce different scores. See [evaluation details](docs/usage.md#evaluation).

## Research materials

This repository contains the extraction and evaluation implementation for *Natural Language Processing based Data Curation and Machine Learning for Polymer Composite Design*. The downstream machine-learning code is maintained in [PolyComp-EMISE-ML](https://github.com/gourav-k/PolyComp-EMISE-ML).

Article PDFs, human annotations, and saved experiment outputs are not included in this repository. Evaluation requires these inputs separately.

## Development

```bash
python -m pip install -e ".[app,dev]"
python -m pytest -q
```

The tests use local fixtures and mocked provider calls; they do not run LLM extraction experiments. Library code is under `src/polycompie/`, the web interface under `app/`, and tests under `tests/`.

## License

Licensed under the [MIT License](LICENSE). Third-party dependencies retain their own licenses.
