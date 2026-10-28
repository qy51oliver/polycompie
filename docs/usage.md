# Usage

Run the following commands from the repository root after installation.
Use only documents you are permitted to process with the selected provider.
Article retrieval and PDF downloads are outside the supplied CLI.

## Credentials

Set the environment variable for the provider used by the batch CLI:

| Provider | Environment variable | CLI model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.2`, `gpt-5.4` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4.6` |
| Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `gemini-2.5-pro` |

The CLI reads environment variables; it does not automatically load a `.env`
file. The web interface accepts an OpenAI key for the current session. Keep
credentials out of source files, commits, and shared logs.

## Prepare a collection

Place each PDF in its own article directory:

```text
data/
└── my_collection/
    └── article_id/
        └── article_id.pdf
```

Initialize a ledger, convert the PDFs with Docling, and clean the Markdown:

```bash
polycompie-ledger --data-dir data --collections my_collection --ledger data/ledger.csv
python -m polycompie.ingestion.pdf_converter --input data/my_collection --ledger data/ledger.csv
python -m polycompie.ingestion.cleaner --input data/my_collection --ledger data/ledger.csv
```

Conversion writes `outputs/docling/<article_id>.md`; cleaning writes
`outputs/cleaned/<article_id>.md`. Both commands update the ledger. Extraction
requires `clean_status=ok` and resolves article paths under `data/` relative to
the working directory.

To use LiteParse, add `--parser liteparse` to both the conversion and cleaning
commands. The web interface automatically tries the other parser if conversion
fails; the batch converter uses only the selected parser.

## Extract samples

```bash
polycompie-extract --article-id article_id --phase 2 --model gpt-5.4
polycompie-extract --batch my_collection --phase 3 --model gpt-5.4
```

Phase 1 targets EMI shielding effectiveness. Phases 2–4 use the 13-property
`selected` list in
[`prompts.yaml`](../src/polycompie/extraction/prompts/prompts.yaml).
The web interface also accepts custom property lists. Ollama is available
through the web interface and Python adapter, not the batch CLI.

For the full CoV workflow, phase 4 without a variant performs its own initial
extraction. Named variants B–E instead reuse variant A's saved extraction:

```bash
polycompie-extract --batch my_collection --phase 4 --model gpt-5.4
# For an ablation with a shared initial extraction:
polycompie-extract --batch my_collection --phase 4 --model gpt-5.4 --variant variant_A
polycompie-extract --batch my_collection --phase 4 --model gpt-5.4 --variant variant_E
```

Outputs are saved under
`data/experiments/<article_id>/phase<N>/<model>/`, with an additional
`variant_<letter>/` directory for named variants. `output.json` contains the
exported samples, `log.json` records the run, and `agentic_logs/` stores
intermediate outputs for agentic workflows. Exported samples exclude evidence
and confidence annotations; those remain available in the traces.

Completed runs are skipped. `--force` reruns extraction and overwrites the
corresponding output. `--seed N` stores a run in an additional `seed_N/`
directory; a seed does not guarantee deterministic provider responses.
`polycompie-extract --help` lists all options.

## Evaluation

Ground truth is read from
`data/annotated_wo_figs/<article_id>/ground_truth.txt`, and predictions from
`data/experiments/<article_id>/phase<N>/<model>/output.json`.
Both contain sample lists; a top-level `samples` wrapper is also accepted.

```bash
polycompie-evaluate --annotated-dir data/annotated_wo_figs --experiments-dir data/experiments --phase 2 --model gpt-5.4 --property-set selected --out-dir data/eval/gpt-5.4-phase2 --llm-model gpt-4o
```

For a named variant, use a model path such as `gpt-5.4/variant_E` with `--phase 4`.
For seeded runs, include the `seed_N` path segment. Property sets are `emi`,
`selected` (13 properties), and `all` (the evaluator's larger predefined list).

The scoring procedure is implemented in
[`evaluate.py`](../src/polycompie/evaluation/evaluate.py) and
[`matching.py`](../src/polycompie/evaluation/matching.py):

- Duplicate composition records are merged separately in ground truth and
  predictions. Ground-truth samples without retained target measurements are
  removed before alignment.
- Matrix matching compares names and descriptions, not matrix composition
  amounts. Filler matching also compares amounts and normalized composition
  types. Both unspecified thicknesses count as agreement; a thickness reported
  on only one side does not.
- Hungarian assignment maximizes total composition similarity. Relaxed and
  strict thresholds are applied to this same assignment. A relaxed match is
  not necessarily a thickness-only disagreement.
- Properties are compared by normalized key, value, and unit within qualifying
  pairs. Known unit conversions are applied. Conditions are not compared.
  Missing values and unsupported units can remove measurements before scoring.
- Missing, unreadable, or empty prediction files are skipped at article level;
  samples unmatched within evaluated articles contribute false positives or
  false negatives. Check article coverage when comparing results.

Outputs include `summary_table.json`, `per_article.json`, `details.json`,
`merge_audit.json`, `llm_match_log.json`, `pr_key_synonyms.json`, and `skips.txt`.
Inspect `per_article.json` for all article exclusions; the summary's skipped
counter and `skips.txt` do not include every empty-input case.

`--no-llm` retains the static synonym and unit tables but disables runtime LLM
matching. Unknown property keys are then dropped. Material judgments are cached
within a run; the CLI does not reload previously saved judgments on a new run.
Preserve the input records, configuration, code revision, and complete evaluation
outputs with any reported metrics.

## Local models

Start an Ollama server and pull a model, then select it in the web interface.
The adapter defaults to `http://localhost:11434`. Local models use phase 2 only.
The raw-response download helps inspect outputs that cannot be parsed as JSON.
