"""Verify the released evaluation offline using current repository source and inputs."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import socket
import sys

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parent
DECISIONS = RESULTS / "replay"


def read(path):
    return json.loads(path.read_bytes())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def no_network(*args, **kwargs):
    raise RuntimeError("Offline replay forbids network connections and fresh judge calls")


def result_documents(E, config, rows):
    """Use the established result filenames and retain exact aggregate counts."""
    counts = E.aggregate_counts(rows)
    metrics = E.make_metrics(counts)
    study, scope = config["study"], config["scope"]
    phase = 1 if study.endswith("single-property") else 2 if study.endswith("multi-property") else 3 if study.endswith("two-step") else 4
    summary = {
        "annotated_dir": "annotations",
        "experiments_dir": "outputs",
        "phase": phase,
        "model": study.split("/")[0],
        "study": study,
        "scope": scope,
        "property_set": config["targets"],
        "evaluated_articles": len(rows),
        "skipped_articles": sum(bool(r.get("skipped")) for r in rows),
        "reference_positive_articles": sum(r["reference_eligible"] for r in rows),
        "prediction_status": config["status"]["prediction_status"],
        "later_saved_prediction_articles": config["status"].get(
            "later_saved_prediction_articles", config["status"]["may_retry_articles"]
        ),
        "avg_similarity": sum(r["avg_similarity"] for r in rows) / len(rows) if rows else 0,
        "composition_matching": metrics["composition"],
        "property_matching": metrics["property"],
        "per_property_metrics": metrics["per_property"],
        "counts": counts,
        "evaluation_policy": "All 59 articles; fixed recorded judgments; no fresh API calls",
    }
    per_article = [{
        "article_id": r["article_id"],
        "n_gt": r["n_gt_samples"],
        "n_pr": r["n_pr_samples"],
        "avg_sim": round(r["avg_similarity"], 3),
        "strict_matches": r["strict_sample_matches"],
        "relaxed_matches": r["relaxed_sample_matches"],
        "skipped": r["skipped"],
        "reference_eligible": r["reference_eligible"],
        "prediction_status": r["prediction_status"],
        "prediction_source": r["prediction_source"],
        "counts": r["_counts"],
    } for r in rows]
    return {"details.json": rows, "per_article.json": per_article, "summary_table.json": summary}


def verify_checksums():
    expected = read(DECISIONS / "checksums.json")
    for category in ["source_files", "input_files", "decision_files"]:
        for relative, checksum in expected[category].items():
            path = ROOT / relative
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"Invalid relative input path: {relative}")
            if sha256(path) != checksum:
                raise ValueError(f"Frozen {category} checksum differs: {relative}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the reproduced files to a new directory")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("The output directory already exists; choose a new directory")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    checksums = verify_checksums()
    sys.path.insert(0, str(ROOT / "src"))
    from polycompie.evaluation import evaluate as E, matching as M
    for module in [E, M]:
        if Path(module.__file__).resolve().parent != ROOT / "src/polycompie/evaluation":
            raise ValueError("Replay must import current repository evaluator source")
    E.OpenAI = no_network
    E._resolve_pr_key_synonyms = no_network
    M._batch_llm_match = no_network

    annotations = {}
    for path in sorted((ROOT / "annotations").glob("*/ground_truth.txt")):
        obj, error = E.parse_gt(path)
        if error:
            raise ValueError(f"{path.parent.name}: {error}")
        annotations[path.parent.name] = E.unwrap_samples(obj)
    if len(annotations) != 59:
        raise ValueError("Replay requires the 59 released annotation files")
    registry = read(DECISIONS / "material_registry.json")
    mapping = read(DECISIONS / "property_mapping.json")
    configurations = read(DECISIONS / "configurations.json")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    article_count = file_count = shortcut_checks = 0
    for config in configurations:
        study, targets = config["study"], config["targets"]
        if config["articles"].keys() != annotations.keys():
            raise ValueError(f"Article coverage differs: {study}")
        local_mapping = {k: v for k, v in mapping.items() if v in targets}
        rows = []
        for aid, article in config["articles"].items():
            prediction = E.unwrap_samples(read(ROOT / "outputs" / study / (aid + ".json")))
            votes, shortcuts = {}, {}
            for identifier in article["decision_ids"]:
                decision = registry[identifier]
                key = tuple(decision[k] for k in ["gt", "pred", "type"])
                if key in votes:
                    raise ValueError(f"Duplicate material decision for {aid}")
                votes[key] = decision["match"]
                if decision["deterministic_ineligible"]:
                    shortcuts[key] = True
            M.reset_caches()
            with redirect_stdout(io.StringIO()):
                row = E.eval_article(aid, annotations[aid], prediction, targets, None, "offline-fixed", local_mapping, material_judgments=votes)
            if row.get("skipped"):
                raise ValueError(f"An article was unexpectedly skipped: {aid}")
            if shortcuts:
                M.reset_caches()
                with redirect_stdout(io.StringIO()):
                    proof = E.eval_article(aid, annotations[aid], prediction, targets, None, "offline-ineligible-proof", local_mapping, material_judgments=votes | shortcuts)
                if proof["_counts"] != row["_counts"] or proof["sample_details"] != row["sample_details"]:
                    raise ValueError(f"An amount-ineligible shortcut affected the result: {aid}")
                shortcut_checks += 1
            row.update(prediction_status="present" if prediction else "empty", prediction_source=article["prediction_source"], reference_eligible=article["reference_eligible"])
            rows.append(row)
            article_count += 1
        directory = Path(config["result_directory"])
        for filename, value in result_documents(E, config, rows).items():
            saved = RESULTS / directory / filename
            if value != read(saved):
                raise ValueError(f"Reproduced result differs: {saved.relative_to(ROOT)}")
            if args.output is not None:
                write(args.output / directory / filename, value)
            file_count += 1
        print(f"Verified {study} ({config['scope']}): 59 articles", flush=True)
    if len(configurations) != 17 or article_count != 1003 or file_count != 51:
        raise ValueError("Unexpected release coverage")
    result = {
        "configurations": len(configurations), "article_evaluations": article_count,
        "exact_result_files": file_count, "source_file_hashes_verified": len(checksums["source_files"]),
        "input_file_hashes_verified": len(checksums["input_files"]),
        "decision_file_hashes_verified": len(checksums["decision_files"]),
        "deterministic_ineligible_invariance_checks": shortcut_checks, "network_calls": 0,
    }
    if args.output is not None:
        write(args.output / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
