"""Review generated pairs, construct dataset splits, and add teacher reasoning.

Run stages with ``python -m data_construction STAGE``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from os.path import normpath
from pathlib import Path

from data_construction.common import TARA_DIR, nonnegative_int, read_json, write_json
from src.data_loader import DATA_ROOT
from src.parsers.cot_tagged import parse
from src.prompts import build_prompt
from src.records import image_path, normalize_record, normalize_type, read_records, validate_pairs

YEAR_CUT = 2018
OOD_TARGET_PER_TYPE = 80
VAL_FRAC = 0.10
SEED = 0


# Review


def review_template(results):
    return [{
        "_id": row["_id"], "accepted": False,
        "claim": row.get("caption", ""), "year": row.get("year"),
        "edit_type": normalize_type(row.get("edit_type")),
        "gt_entity_fine": "", "gt_entity_canonical": "", "gt_why_contradicts": "",
        "proposed_change": row.get("what_changed", ""),
        "proposed_reason": row.get("why_contradicts", ""),
        "original_image": row["local_image"],
        "edited_image": row["edit_status"]["output_image"],
        "review_notes": "",
    } for row in results if row.get("edit_status", {}).get("output_image")
        and not row["edit_status"].get("error")]


def assemble(results, reviews, data_root, original_dir="TARA/images", edited_dir="TARA/edited_images"):
    edits = {}
    for row in results:
        if row["_id"] in edits:
            raise ValueError(f"Duplicate edit result: {row['_id']}")
        edits[row["_id"]] = row
    annotations = []
    reviewed_ids = set()
    for review in reviews:
        if review["_id"] in reviewed_ids:
            raise ValueError(f"Duplicate review: {review['_id']}")
        reviewed_ids.add(review["_id"])
        if review["_id"] not in edits:
            raise ValueError(f"Review has no matching edit result: {review['_id']}")
        if review.get("accepted") is not True:
            continue
        row = edits[review["_id"]]
        if review.get("claim") != row.get("caption"):
            raise ValueError(f"{row['_id']}: review must preserve the source claim")
        status = row.get("edit_status", {})
        if status.get("error") or not status.get("output_image"):
            raise ValueError(f"Approved edit has no successful generation: {row['_id']}")
        if (review.get("original_image") != row.get("local_image")
                or review.get("edited_image") != status["output_image"]):
            raise ValueError(f"{row['_id']}: review images differ from the current generation; review the current pair")
        entity = review.get("gt_entity_canonical")
        if type(review.get("year")) is not int or not isinstance(entity, str) or not entity.strip():
            raise ValueError(f"{row['_id']}: review requires year and gt_entity_canonical")
        shared = {key: review.get(key) for key in (
            "_id", "claim", "year", "edit_type", "gt_entity_fine", "gt_entity_canonical", "gt_why_contradicts")}
        shared["editing_prompt"] = row.get("editing_prompt")
        shared["gt_entity_canonical"] = entity.strip()
        for image_type, local, generator in (
            ("original", str(Path(original_dir) / row["local_image"]), "none"),
            ("edited", str(Path(edited_dir) / status["output_image"]), row.get("generator", "unknown")),
        ):
            path = image_path(data_root, local)
            if not path.is_file():
                raise FileNotFoundError(path)
            annotations.append(normalize_record(dict(shared, image_type=image_type, local_image=local, generator=generator)))
    if not annotations:
        raise ValueError("No approved pairs; complete the review file before assembling annotations")
    validate_pairs(annotations)
    return annotations


def review_main(argv=None):
    p = argparse.ArgumentParser(description='Create review templates or assemble approved original/edited image pairs.')
    p.add_argument("--edit-results", type=Path, default=TARA_DIR / "edit_results.json")
    p.add_argument("--reviews", type=Path)
    p.add_argument("--export-review", type=Path, help="Write a template for human review and exit")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--original-dir", default="TARA/images")
    p.add_argument("--edited-dir", default="TARA/edited_images")
    p.add_argument("--output", type=Path, default=DATA_ROOT / "annotations.json")
    args = p.parse_args(argv)
    results = read_json(args.edit_results)
    if args.export_review:
        out, rows = args.export_review, review_template(results)
        if out.exists():
            p.error(f"Refusing to overwrite existing review: {out}")
    else:
        if not args.reviews:
            p.error("Provide --reviews or --export-review")
        out = args.output
        rows = assemble(results, read_json(args.reviews), args.data_root, args.original_dir, args.edited_dir)
    write_json(out, rows)
    print(f"Wrote {len(rows)} records: {out}")


# Split


def year_bucket(year):
    """Fixed reporting/stratification buckets, independent of the temporal cut."""
    if year < 2010:
        return "lt2010"
    if year < 2014:
        return "2010_13"
    if year < 2018:
        return "2014_17"
    return "ge2018"


def build_splits(records, *, year_cut=YEAR_CUT, ood_target=OOD_TARGET_PER_TYPE,
                 val_frac=VAL_FRAC, seed=SEED, source_name="annotations.json"):
    """Return split rows, discard/OOD manifests, and a report without writing files.

    ``records`` should contain normalized annotations (as returned by
    ``read_records``). Sorting pairs and using a local RNG makes results
    independent of input order and leaves the caller's records unchanged.
    """
    if not 0 <= val_frac < 1 or ood_target < 1:
        raise ValueError("val_frac must be in [0, 1) and ood_target must be positive")
    rng = random.Random(seed)
    pairs = []
    entity_total = Counter()
    entity_recent = Counter()
    entity_type = {}
    for article_id, pair in sorted(validate_pairs(records).items()):
        edited, original = pair["edited"], pair["original"]
        entity = edited.get("gt_entity_canonical")
        if not isinstance(entity, str) or not entity.strip() or type(edited.get("year")) is not int:
            raise ValueError(f"{article_id}: entity splitting requires gt_entity_canonical and integer year")
        if entity in entity_type and entity_type[entity] != edited["edit_type"]:
            raise ValueError(f"Canonical entity {entity!r} belongs to multiple edit types")
        entity_type[entity] = edited["edit_type"]
        entity_total[entity] += 1
        entity_recent[entity] += edited["year"] >= year_cut
        pairs.append((article_id, edited, original))

    # Prefer entities concentrated after the cutoff, with deterministic ties.
    ood_entities = set()
    ood_picks = {}
    for edit_type in sorted(set(entity_type.values())):
        candidates = [(entity, entity_recent[entity], entity_total[entity])
                      for entity, typ in entity_type.items()
                      if typ == edit_type and entity_recent[entity] >= 1]
        candidates.sort(key=lambda x: (-x[1] / x[2], x[2] - x[1], -x[1], x[0]))
        picked = []
        recent_count = 0
        for entity, recent, total in candidates:
            picked.append((entity, recent, total))
            ood_entities.add(entity)
            recent_count += recent
            if recent_count >= ood_target:
                break
        ood_picks[edit_type] = picked

    all_recent_entities = {entity for entity in entity_type
                           if entity_recent[entity] == entity_total[entity]
                           and entity not in ood_entities}
    train_pool, test_id_pool, test_ood, discarded = [], [], [], []
    for pair in pairs:
        _, edited, _ = pair
        entity = edited["gt_entity_canonical"]
        recent = edited["year"] >= year_cut
        if entity in ood_entities:
            (test_ood if recent else discarded).append(pair)
        elif entity in all_recent_entities:
            discarded.append(pair)
        else:
            (test_id_pool if recent else train_pool).append(pair)

    # Preserve insertion order here: it controls the sequence of RNG shuffles.
    ood_counts = Counter(edited["edit_type"] for _, edited, _ in test_ood)
    id_by_type = defaultdict(list)
    for pair in test_id_pool:
        id_by_type[pair[1]["edit_type"]].append(pair)
    test_id = []
    for edit_type, items in id_by_type.items():
        rng.shuffle(items)
        keep = ood_counts[edit_type]
        test_id.extend(items[:keep])
        discarded.extend(items[keep:])

    strata = defaultdict(list)
    for pair in train_pool:
        _, edited, _ = pair
        key = (edited["edit_type"], edited["gt_entity_canonical"], year_bucket(edited["year"]))
        strata[key].append(pair)
    train, val = [], []
    for items in strata.values():
        rng.shuffle(items)
        n_val = (min(len(items) - 1, max(1, int(round(len(items) * val_frac))))
                 if len(items) >= 2 and val_frac > 0 else 0)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    if not train:
        raise ValueError("No training pairs remain; supply more pre-cut ID examples or reduce OOD selection")

    def discard_reason(edited):
        entity = edited["gt_entity_canonical"]
        if entity in ood_entities and edited["year"] < year_cut:
            return "ood_pre_cut"
        if entity in all_recent_entities:
            return "all_recent_no_train_anchor"
        return "id_recent_surplus"

    split_items = {"train": train, "val": val, "test_id_edit": test_id, "test_ood_edit": test_ood}
    image_owners = {}
    for name, items in split_items.items():
        for article_id, _, original in items:
            path = normpath(original["local_image"])
            previous = image_owners.setdefault(path, (name, article_id))
            if previous[0] != name:
                raise ValueError(f"Source image overlaps {previous[0]} and {name}: {path} "
                                 f"(claims {previous[1]} and {article_id}); resolve duplicate source images before splitting")
    result = {}
    for name, items in split_items.items():
        rows = []
        for _, edited, original in items:
            rows.append(dict(original, year=edited["year"], edit_type=edited["edit_type"]))
            rows.append(dict(edited))
        result[name] = rows
    result["discarded"] = [
        {"stem": stem, "edit_type": edited["edit_type"], "entity": edited["gt_entity_canonical"],
         "year": edited["year"], "reason": discard_reason(edited)}
        for stem, edited, _ in discarded
    ]
    result["ood_entities"] = {typ: [entity for entity, _, _ in picks] for typ, picks in ood_picks.items()}
    result["report"] = _build_report(
        dict(split_items, discard=discarded), ood_picks,
        source_name=source_name, year_cut=year_cut, ood_target=ood_target, val_frac=val_frac, seed=seed,
    )
    return result


def _build_report(split_items, ood_picks, *, source_name, year_cut, ood_target, val_frac, seed):
    lines = ["# Splits Report (entity + temporal)", "",
             f"- Source: `{source_name}`",
             f"- year_cut: **{year_cut}**  (train/val < {year_cut}, test >= {year_cut})",
             f"- ood_target_per_type: **{ood_target}**",
             f"- val_frac: **{val_frac}**",
             f"- seed: {seed}", "", "## Split sizes (pair-doubled)"]
    for name, items in split_items.items():
        label = "DISCARD (not used)" if name == "discard" else name
        lines.append(f"- **{label}**: {len(items)} pairs ({2 * len(items)} samples)")
    lines += ["", "## Per-edit_type counts (edited only)",
              "| edit_type | train | val | test_id | test_ood | discard |",
              "|---|---:|---:|---:|---:|---:|"]
    counts = [Counter(edited["edit_type"] for _, edited, _ in items) for items in split_items.values()]
    for edit_type in ood_picks:
        lines.append("| " + " | ".join([edit_type] + [str(count[edit_type]) for count in counts]) + " |")
    lines += ["| **TOTAL** | " + " | ".join(str(sum(count.values())) for count in counts) + " |",
              "", "## OOD entities per type"]
    for edit_type, picks in ood_picks.items():
        lines.append(f"\n### {edit_type}  ({len(picks)} entities, {sum(n for _, n, _ in picks)} year≥{year_cut} samples)\n")
        for entity, recent, total in picks:
            lines.append(f"- `{entity}`  ({recent} of {total} samples are year≥{year_cut}, ratio={recent/total*100:.0f}%)")
    lines += ["", "## Year-bucket distribution per split (edited only)",
              "| split | <2010 | 2010-13 | 2014-17 | ≥2018 | total |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, items in split_items.items():
        if name == "discard":
            continue
        counts = Counter(year_bucket(edited["year"]) for _, edited, _ in items)
        lines.append(f"| {name} | {counts['lt2010']} | {counts['2010_13']} | {counts['2014_17']} | {counts['ge2018']} | {sum(counts.values())} |")

    entities = {name: {edited["gt_entity_canonical"] for _, edited, _ in items}
                for name, items in split_items.items()}
    orphan_entities = entities["test_id_edit"] - entities["train"]
    leak_entities = entities["test_ood_edit"] & entities["train"]
    train_years = [edited["year"] for _, edited, _ in split_items["train"]]
    test_years = [edited["year"] for name in ("test_id_edit", "test_ood_edit")
                  for _, edited, _ in split_items[name]]
    lines += ["", "## Split validation",
              f"- unique entities: train={len(entities['train'])}, test_id={len(entities['test_id_edit'])}, test_ood={len(entities['test_ood_edit'])}",
              f"- ID test entities absent from training: {len(orphan_entities)} (expected: 0)",
              f"- OOD test entities present in training: {len(leak_entities)} (expected: 0)",
              f"- train year range: {min(train_years)}-{max(train_years)}; min test year: {min(test_years, default=None)}", ""]
    if orphan_entities:
        lines.append(f"  - ID entities without training examples: {sorted(orphan_entities)[:10]}")
    return "\n".join(lines)


def split_main(argv=None):
    parser = argparse.ArgumentParser(description='Build paired entity + temporal splits without claim leakage.')
    parser.add_argument("--year-cut", "--year_cut", type=int, default=YEAR_CUT)
    parser.add_argument("--ood-target", "--ood_target", type=int, default=OOD_TARGET_PER_TYPE)
    parser.add_argument("--val-frac", "--val_frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--annotations", type=Path, default=DATA_ROOT / "annotations.json")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "splits")
    args = parser.parse_args(argv)
    if not 0 <= args.val_frac < 1 or args.ood_target < 1:
        parser.error("val_frac must be in [0, 1) and ood_target must be positive")
    result = build_splits(
        read_records(args.annotations), year_cut=args.year_cut, ood_target=args.ood_target,
        val_frac=args.val_frac, seed=args.seed, source_name=args.annotations.name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        filename = "split_report.md" if name == "report" else f"{name}.json"
        content = value if name == "report" else json.dumps(value, ensure_ascii=False, indent=2)
        (args.output_dir / filename).write_text(content, encoding="utf-8")

    counts = {name: len(result[name]) // 2 for name in ("train", "val", "test_id_edit", "test_ood_edit")}
    entities = {name: {row["gt_entity_canonical"] for row in result[name][1::2]} for name in counts}
    print(f"wrote splits to {args.output_dir}")
    print(f"  train       : {counts['train']} pairs ({2 * counts['train']} samples)")
    print(f"  val         : {counts['val']} pairs")
    print(f"  test_id_edit: {counts['test_id_edit']} pairs")
    print(f"  test_ood_edit:{counts['test_ood_edit']} pairs")
    print(f"  discarded   : {len(result['discarded'])} pairs")
    print(f"  OOD test entities present in training: {len(entities['test_ood_edit'] & entities['train'])}")
    print(f"  ID test entities absent from training: {len(entities['test_id_edit'] - entities['train'])}")


# Annotate


def annotate_main(argv=None):
    p = argparse.ArgumentParser(description='Generate optional teacher reasoning for approved train/validation records.')
    p.add_argument("--input", type=Path, required=True, help="A reviewed train or validation JSON file")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--model", default="gpt-5.5", help="Teacher model identifier (paper: gpt-5.5)")
    p.add_argument("--limit", type=nonnegative_int)
    p.add_argument("--resume", action="store_true", help="Continue an existing checkpoint with the same source labels and teacher")
    args = p.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        p.error("Choose a separate output file so reviewed source records are preserved")
    if args.output.exists() and not args.resume:
        p.error("Choose a new output file or use --resume to continue an existing annotation checkpoint")
    rows = read_records(args.input, args.data_root, check_images=True)

    def key(row):
        return row["_id"], row["image_type"]

    def labels_match(row, raw):
        parsed = parse(raw)
        edited = row["image_type"] == "edited"
        return (parsed["parse_ok"] and parsed["verdict"] == ("INCONSISTENT" if edited else "CONSISTENT")
                and parsed["type"] == (row["edit_type"] if edited else None))

    sources = {key(row): row for row in rows}
    completed = {}
    if args.resume and args.output.exists():
        saved = read_json(args.output)
        if not isinstance(saved, list):
            raise ValueError("Annotation checkpoint must be a JSON array")
        for row in saved:
            identity = key(row)
            source = sources.get(identity)
            if identity in completed or source is None:
                raise ValueError(f"Annotation checkpoint has duplicate or unknown sample: {identity}")
            if (row.get("teacher_model") != args.model
                    or any(row.get(field) != value for field, value in source.items()
                           if field not in {"cot_response", "teacher_model"})
                    or not isinstance(row.get("cot_response"), str)
                    or not labels_match(source, row["cot_response"])):
                raise ValueError(f"Annotation checkpoint differs from source labels or teacher: {identity}")
            completed[identity] = row

    def save():
        write_json(args.output, [completed[key(row)] for row in rows if key(row) in completed])

    pending = [row for row in rows if key(row) not in completed][:args.limit]
    if not pending:
        save()
        print(f"Annotated {len(completed)} records: {args.output}")
        return
    from src.backends.api_backend import APIBackend

    backend = APIBackend("openai", args.model)
    backend.load()
    try:
        for row in pending:
            expected = "INCONSISTENT" if row["image_type"] == "edited" else "CONSISTENT"
            context = json.dumps({k: row.get(k) for k in ("edit_type", "gt_entity_fine", "gt_why_contradicts")})
            prompt = (build_prompt(row["claim"]) + "\nReviewed training labels: " + context +
                      f"\nReviewed verdict: {expected}. Use these as supervision, but ground each observation in the image.")
            raw = backend.run(str(image_path(args.data_root, row["local_image"])), prompt)
            if not labels_match(row, raw):
                raise ValueError(f"Teacher response disagrees with reviewed labels: {row['_id']}")
            completed[key(row)] = dict(row, cot_response=raw, teacher_model=args.model)
            save()
    finally:
        backend.unload()
    print(f"Annotated {len(completed)} records: {args.output}")
