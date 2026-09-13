"""Prepare source images, screen visible cues, and generate contextual edits.

Run stages with ``python -m data_construction STAGE``.
"""

import argparse
import base64
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PIL import Image

from data_construction.common import (
    IMAGE_DIR, TARA_DIR, add_batch_arguments, chat_json, metadata,
    nonnegative_int, positive_int, read_json, run_api_batch, write_bytes, write_json,
)
from data_construction.prompts import (
    EDIT_PROMPT_TEMPLATE, FILTER_PROMPT, INCONSISTENCY_DEFS,
    INCONSISTENCY_TYPES, REGION_POOLS, STRICT_PROMPT,
)

SUCCESS_STATUSES = {"ok_jumbo", "ok_fallback", "skipped"}


def valid_image(source):
    """Check decoded image content instead of accepting HTTP error pages or partial files."""
    try:
        with Image.open(source) as image:
            image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def validate_stage_result(result, result_key):
    """Reject malformed model JSON so it is checkpointed as a retryable failure."""
    if not isinstance(result, dict) or "error" in result:
        raise ValueError(f"Invalid {result_key} response")
    feasible = result.get("feasible_types")
    if result_key == "filter_result":
        if (type(result.get("suitable")) is not bool or not isinstance(feasible, list)
                or any(not isinstance(name, str) or name not in INCONSISTENCY_DEFS for name in feasible)
                or (result["suitable"] and not feasible)):
            raise ValueError("Screening response requires boolean suitable and known feasible_types")
    elif result_key == "round2_result":
        if (result.get("quality") not in {"high", "medium", "low"} or not isinstance(feasible, dict)
                or any(name not in INCONSISTENCY_DEFS or not isinstance(info, dict)
                       or type(info.get("visible")) is not bool for name, info in feasible.items())):
            raise ValueError("Verification response requires quality and boolean visibility for known types")
    elif result_key == "edit_result":
        if not isinstance(result.get("editing_prompt"), str) or not result["editing_prompt"].strip():
            raise ValueError("Proposal response requires a nonempty editing_prompt")
    return result


def stage_completed(row, result_key):
    try:
        validate_stage_result(row.get(result_key), result_key)
        return True
    except (ValueError, TypeError):
        return False


# Prepare


def prepare(paths, claim_field="caption"):
    records = {}
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
                if path.suffix == ".jsonl" else json.loads(text))
        if not isinstance(rows, list):
            raise ValueError(f"Expected records in a JSON array or JSONL: {path}")
        for raw in rows:
            row = dict(raw)
            for key in ("_id", "image_url", claim_field):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}: missing {key}; use --claim-field for the original caption field")
            row["caption"] = row[claim_field].strip()
            if row["_id"] in records:
                old = records[row["_id"]]
                if (old["image_url"], old["caption"]) != (row["image_url"], row["caption"]):
                    raise ValueError(f"Conflicting source records for {row['_id']}")
            else:
                records[row["_id"]] = row
    return list(records.values())


def prepare_main(argv=None):
    p = argparse.ArgumentParser(description='Convert downloaded TARA JSON/JSONL metadata into image-download records.')
    p.add_argument("--input", nargs="+", type=Path, required=True)
    p.add_argument("--claim-field", default="caption")
    p.add_argument("--output", type=Path, default=TARA_DIR / "filtered_metadata.json")
    args = p.parse_args(argv)
    records = prepare(args.input, args.claim_field)
    write_json(args.output, records)
    print(f"Prepared {len(records)} unique claims: {args.output}")


# Download


def url_to_filename(url):
    """Convert URL to a safe filename using hash + original extension."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    name_part = Path(urlsplit(url).path).name or "image"
    return f"{url_hash}_{name_part}"


def get_jumbo_url(url):
    """Convert articleLarge URL to jumbo URL for higher resolution."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=parts.path.replace("articleLarge", "jumbo"), fragment=""))


def download_one(record, image_dir, timeout=15):
    """Return (record ID, filename, status), trying jumbo before the original."""
    import requests

    url = record["image_url"]
    jumbo_url = get_jumbo_url(url)
    jumbo_filename = f"{hashlib.md5(jumbo_url.encode()).hexdigest()[:12]}_jumbo.jpg"
    original_filename = url_to_filename(url)
    candidates = [
        (jumbo_url, jumbo_filename, "ok_jumbo"),
        (url, original_filename, "ok_fallback"),
    ]
    image_dir = Path(image_dir)

    for _, filename, _ in candidates:
        path = image_dir / filename
        if path.is_file() and valid_image(path):
            return record["_id"], filename, "skipped"

    status = "failed"
    for download_url, filename, success_status in candidates:
        try:
            response = requests.get(download_url, timeout=timeout)
            if response.status_code == 200 and valid_image(io.BytesIO(response.content)):
                write_bytes(image_dir / filename, response.content)
                return record["_id"], filename, success_status
            status = "failed"
        except (requests.RequestException, OSError) as error:
            status = f"error: {error}"

    return record["_id"], original_filename, status


def format_stats(stats):
    return (
        f"jumbo={stats['ok_jumbo']} fallback={stats['ok_fallback']} "
        f"skipped={stats['skipped']} failed={stats['failed']}"
    )


def download_main(argv=None):
    parser = argparse.ArgumentParser(description='Download TARA images, preferring jumbo resolution and reusing cached files.')
    parser.add_argument("--metadata", type=Path, default=TARA_DIR / "filtered_metadata.json")
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=TARA_DIR / "filtered_metadata_with_images.json")
    parser.add_argument("--max-workers", "--max_workers", type=positive_int, default=8)
    parser.add_argument("--limit", type=nonnegative_int)
    args = parser.parse_args(argv)

    records = read_json(args.metadata)
    if args.limit is not None:
        records = records[:args.limit]
    args.image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(records)} images to {args.image_dir} ({args.max_workers} workers)")
    stats = Counter()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_one, record, args.image_dir): record for record in records}
        for done, future in enumerate(as_completed(futures), start=1):
            _, filename, status = future.result()
            record = futures[future]
            if status in SUCCESS_STATUSES:
                record["local_image"] = filename
                stats[status] += 1
            else:
                record.pop("local_image", None)
                stats["failed"] += 1
            if done % 200 == 0 or done == len(records):
                print(f"  [{done}/{len(records)}] {format_stats(stats)}")

    write_json(args.output, records)
    print(f"Done! {format_stats(stats)}")
    print(f"Updated metadata saved to: {args.output}")


# Filter


def filter_one(client, record, image_dir, model="gpt-4o-mini", min_width=1024, min_height=1024):
    """Return the claim ID and screening result, including per-image errors."""
    image_path = Path(image_dir) / record.get("local_image", "")
    if not image_path.is_file():
        return record["_id"], {"error": "image_not_found"}
    try:
        with Image.open(image_path) as image:
            if image.width < min_width or image.height < min_height:
                return record["_id"], {"error": f"low_resolution_{image.width}x{image.height}px"}
    except OSError:
        return record["_id"], {"error": "image_read_error"}

    fields = metadata(record)
    prompt = FILTER_PROMPT.format(
        **{key: fields[key] for key in ("caption", "headline")},
        location=fields["location"] or "unknown", time=fields["time"] or "unknown",
        keywords=", ".join(fields["keywords"][:8]), type_descs=INCONSISTENCY_TYPES,
    )
    try:
        result = validate_stage_result(
            chat_json(client, model, prompt, image_path=image_path, detail="low", max_tokens=400), "filter_result")
        return record["_id"], result
    except Exception as exc:
        return record["_id"], {"error": str(exc)}


def filter_main(argv=None):
    parser = argparse.ArgumentParser(description='Screen source image–claim pairs for feasible contextual edits.')
    add_batch_arguments(parser, workers=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--min-width", "--min_width", type=nonnegative_int, default=1024)
    parser.add_argument("--min-height", "--min_height", type=nonnegative_int, default=1024)
    parser.add_argument("--metadata", type=Path, default=TARA_DIR / "filtered_metadata_with_images.json")
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=TARA_DIR / "filter_results.json")
    args = parser.parse_args(argv)
    records = [row for row in read_json(args.metadata) if row.get("local_image")]

    def process(row, client):
        _, result = filter_one(client, row, args.image_dir, args.model, args.min_width, args.min_height)
        return dict(metadata(row), filter_result=result)

    run_api_batch(records, process, args.output, result_key="filter_result",
                  workers=args.max_workers, limit=args.limit, resume=args.resume,
                  completed=lambda row: stage_completed(row, "filter_result"))


# Verify


def verify_one(client, item, image_dir, model="gpt-5.5"):
    image_path = Path(image_dir) / item.get("local_image", "")
    if not image_path.is_file():
        return item["_id"], {"error": "image_not_found"}
    fields = metadata(item)
    prompt = STRICT_PROMPT.format(caption=fields["caption"], location=fields["location"], time=fields["time"])
    try:
        return item["_id"], validate_stage_result(chat_json(client, model, prompt, image_path=image_path), "round2_result")
    except Exception as exc:
        return item["_id"], {"error": str(exc)}


def verify_main(argv=None):
    parser = argparse.ArgumentParser(description='Verify clearly visible edit targets in pairs that passed the first screen.')
    add_batch_arguments(parser, workers=16)
    parser.add_argument("--model", default="gpt-5.5", help="Vision model used for strict verification (paper: gpt-5.5)")
    parser.add_argument("--round1-path", "--round1_path", type=Path, default=TARA_DIR / "filter_results.json")
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", "--output-path", "--output_path", type=Path, default=TARA_DIR / "filter_round2_results.json")
    args = parser.parse_args(argv)
    items = [row for row in read_json(args.round1_path)
             if row.get("filter_result", {}).get("suitable") is True and "error" not in row["filter_result"]]

    def process(row, client):
        _, result = verify_one(client, row, args.image_dir, args.model)
        return dict(metadata(row), round1_types=row["filter_result"].get("feasible_types", []), round2_result=result)

    run_api_batch(items, process, args.output, result_key="round2_result",
                  workers=args.max_workers, limit=args.limit, resume=args.resume,
                  completed=lambda row: stage_completed(row, "round2_result"))


# Propose


def get_visible_types(item):
    feasible = item.get("round2_result", {}).get("feasible_types", {})
    if not isinstance(feasible, dict):
        return []
    return [name for name, info in feasible.items()
            if isinstance(info, dict) and info.get("visible") is True and name in INCONSISTENCY_DEFS]


def get_evidence(item, type_name):
    info = item.get("round2_result", {}).get("feasible_types", {}).get(type_name, {})
    return info.get("evidence", "") if isinstance(info, dict) else ""


def assign_types(data):
    """Process constrained images first, selecting the least assigned visible type."""
    assignments, counts = {}, Counter()
    for item in sorted(data, key=lambda row: (len(get_visible_types(row)), row["_id"])):
        visible = get_visible_types(item)
        if visible:
            chosen = min(visible, key=lambda name: (counts[name], name))
            assignments[item["_id"]] = chosen
            counts[chosen] += 1
    return assignments


def generate_one(client, item, edit_type, model="gpt-5.5", region_idx=0):
    pool = REGION_POOLS.get(edit_type, ["any region"])
    region_hint = pool[region_idx % len(pool)]
    prompt = EDIT_PROMPT_TEMPLATE.format(
        caption=item.get("caption", ""), headline=item.get("headline", ""),
        keywords=", ".join(item.get("keywords", [])[:6]),
        visual_description=item.get("round2_result", {}).get("visual_description", "Not available."),
        edit_type=edit_type, evidence=get_evidence(item, edit_type),
        inconsistency_def=INCONSISTENCY_DEFS[edit_type], region_hint=region_hint,
    )
    try:
        result = validate_stage_result(chat_json(client, model, prompt, temperature=0.7), "edit_result")
        return item["_id"], dict(result, region_hint=region_hint)
    except Exception as exc:
        return item["_id"], {"error": str(exc)}


def propose_main(argv=None):
    parser = argparse.ArgumentParser(description='Assign balanced edit types and propose one contextual edit per verified image.')
    add_batch_arguments(parser, workers=16)
    parser.add_argument("--model", default="gpt-5.5", help="Text model used to propose edits (paper: gpt-5.5)")
    parser.add_argument("--output", type=Path, default=TARA_DIR / "edit_prompts.json")
    parser.add_argument("--round2-path", "--round2_path", type=Path, default=TARA_DIR / "filter_round2_results.json")
    parser.add_argument("--quality", choices=["high", "high+medium", "all"], default="high")
    args = parser.parse_args(argv)
    data = [row for row in read_json(args.round2_path)
            if "error" not in row.get("round2_result", {}) and
            (args.quality == "all" or row.get("round2_result", {}).get("quality") in args.quality.split("+"))]
    assignments = assign_types(data)
    counters, regions = Counter(), {}
    items = [row for row in sorted(data, key=lambda row: row["_id"]) if row["_id"] in assignments]
    # Assign across the full candidate pool before applying resume or limit.
    for row in items:
        edit_type = assignments[row["_id"]]
        regions[row["_id"]] = counters[edit_type]
        counters[edit_type] += 1

    def process(row, client):
        edit_type = assignments[row["_id"]]
        _, result = generate_one(client, row, edit_type, args.model, regions[row["_id"]])
        return dict(row, prompt_model=args.model, edit_type=edit_type, evidence=get_evidence(row, edit_type),
                    all_visible_types=get_visible_types(row), edit_result=result)

    run_api_batch(items, process, args.output, result_key="edit_result",
                  workers=args.max_workers, limit=args.limit, resume=args.resume,
                  completed=lambda row: stage_completed(row, "edit_result"))


# Edit


def edit_one(client, item, image_dir, output_dir, model="gpt-image-1.5", quality="medium"):
    """Return the claim ID and generation status; each claim gets its own file."""
    rec_id = item["_id"]
    image_path = Path(image_dir) / item.get("local_image", "")
    if not image_path.is_file():
        return rec_id, {"error": "image_not_found", "image_path": str(image_path)}
    editing_prompt = item.get("edit_result", {}).get("editing_prompt", "")
    if not editing_prompt:
        return rec_id, {"error": "no_editing_prompt"}

    suffix = hashlib.sha256(str(rec_id).encode()).hexdigest()[:12]
    try:
        with image_path.open("rb") as image:
            response = client.images.edit(model=model, image=image, prompt=editing_prompt, quality=quality, size="auto")
        image_bytes = base64.b64decode(response.data[0].b64_json, validate=True)
        if not valid_image(io.BytesIO(image_bytes)):
            raise ValueError("Image generation returned invalid image bytes")
        content_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
        output_filename = f"{image_path.stem}_{suffix}_{content_hash}_edited.png"
        output_path = Path(output_dir) / output_filename
        write_bytes(output_path, image_bytes)
        return rec_id, {"output_image": output_filename, "size_bytes": len(image_bytes)}
    except Exception as exc:
        message = str(exc)
        if "moderation" in message.lower() or "safety" in message.lower():
            return rec_id, {"error": "moderation_blocked", "detail": message[:200]}
        return rec_id, {"error": message[:300]}


def edit_main(argv=None):
    parser = argparse.ArgumentParser(description='Apply contextual edit prompts and save generated images for human review.')
    add_batch_arguments(parser, workers=10)
    parser.add_argument("--model", default="gpt-image-1.5")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--prompts", type=Path, default=TARA_DIR / "edit_prompts.json")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=TARA_DIR / "edited_images")
    parser.add_argument("--results", type=Path, default=TARA_DIR / "edit_results.json")
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=IMAGE_DIR)
    args = parser.parse_args(argv)

    def completed(row):
        status = row.get("edit_status", {})
        return ("error" not in status and bool(status.get("output_image"))
                and (args.output_dir / status["output_image"]).is_file()
                and valid_image(args.output_dir / status["output_image"]))

    def process(row, client):
        _, result = edit_one(client, row, args.image_dir, args.output_dir, args.model, args.quality)
        fields = {key: row.get("edit_result", {}).get(key, "") for key in (
            "editing_prompt", "what_changed", "why_contradicts", "world_knowledge_needed", "difficulty", "region_hint")}
        return dict(row, **fields, generator=args.model, generation_quality=args.quality, edit_status=result)

    run_api_batch(read_json(args.prompts), process, args.results, result_key="edit_status",
                  workers=args.max_workers, limit=args.limit, resume=args.resume, completed=completed)
