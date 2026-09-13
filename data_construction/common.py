"""Shared I/O, metadata, and batch execution for construction stages."""

import argparse
import base64
import json
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.data_loader import DATA_ROOT

TARA_DIR = DATA_ROOT / "TARA"
IMAGE_DIR = TARA_DIR / "images"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace a JSON checkpoint atomically, including on the first write."""
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_bytes(path, value):
    """Publish a complete file; interrupted writes must not become cached successes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(value)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_int(value):
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def add_batch_arguments(parser, workers):
    parser.add_argument("--max-workers", "--max_workers", type=positive_int, default=workers)
    parser.add_argument("--limit", type=nonnegative_int, help="Maximum pending records; 0 makes no requests")
    parser.add_argument("--resume", action="store_true", help="Keep successes and retry failed records")


def create_client():
    from openai import OpenAI
    from src.environment import load_environment

    load_environment()
    return OpenAI()


def image_to_base64_uri(path):
    from PIL import Image

    path = Path(path)
    with Image.open(path) as image:
        media = Image.MIME[image.format]
    return f"data:{media};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def chat_json(client, model, prompt, *, image_path=None, detail="high", max_tokens=500, temperature=0):
    """Request one JSON object, retaining the stages' model-specific settings."""
    content = prompt
    if image_path is not None:
        content = [
            {"type": "image_url", "image_url": {"url": image_to_base64_uri(image_path), "detail": detail}},
            {"type": "text", "text": prompt},
        ]
    reasoning = model.startswith("gpt-5")
    params = {"max_completion_tokens": 16000} if reasoning else {
        "max_tokens": max_tokens, "temperature": temperature,
    }
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"}, **params,
    )
    result = json.loads(response.choices[0].message.content)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the model")
    if response.usage is not None:
        result["tokens"] = {"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}
    return result


def metadata(record):
    """Normalize TARA metadata, deriving dates from the URL only when missing."""
    headline = record.get("headline") or ""
    keywords = record.get("keywords") or []
    match = re.search(r"/images/(\d{4})/(\d{2})/(\d{2})/", record.get("image_url", ""))
    date = record.get("date") or ("-".join(match.groups()) if match else "")
    return {
        "_id": record["_id"], "caption": record.get("caption", ""),
        "headline": headline.get("main", "") if isinstance(headline, dict) else headline,
        "location": record.get("gold_location_suggest") or record.get("location", ""),
        "time": (record.get("gold_time_suggest") or record.get("time") or record.get("date")
                 or record.get("pub_date") or date or record.get("year") or ""),
        "year": record.get("year") or (match.group(1) if match else ""),
        "date": date, "local_image": record.get("local_image", ""),
        "image_url": record.get("image_url", ""),
        "keywords": [word.get("value", "") if isinstance(word, dict) else word for word in keywords],
    }


def run_api_batch(records, process, output, *, result_key, workers=8, limit=None, resume=False,
                  completed=None, save_every=50):
    """Process records with one client and keep one checkpoint row per claim.

    ``process(record, client)`` returns the complete output row. A completed
    stage result has no error; image stages can additionally check their files.
    Selection happens before limit, and results are saved in stable ID order.
    """
    if workers < 1 or save_every < 1 or (limit is not None and limit < 0):
        raise ValueError("workers/save_every must be positive and limit nonnegative")
    records = list(records)
    if len({row["_id"] for row in records}) != len(records):
        raise ValueError("Duplicate source record IDs")
    output = Path(output)
    results = {row["_id"]: row for row in read_json(output)} if resume and output.exists() else {}
    unknown = set(results) - {row["_id"] for row in records}
    if unknown:
        raise ValueError("Checkpoint contains records outside the current source; use a new output file")
    if completed is None:
        completed = lambda row: isinstance(row.get(result_key), dict) and "error" not in row[result_key]
    pending = [row for row in records if row["_id"] not in results or not completed(results[row["_id"]])]
    pending = pending[:limit]

    def save():
        write_json(output, [results[key] for key in sorted(results)])

    if not pending:
        if not output.exists() or not resume:
            save()
        print("Nothing to process.")
        return list(results.values())

    def collect(future, row):
        try:
            result = future.result()
        except Exception as exc:
            result = dict(row, **{result_key: {"error": str(exc)}})
        results[row["_id"]] = result

    client = create_client()
    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, row, client): row for row in pending}
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    collect(future, futures[future])
                    if done % save_every == 0 or done == len(pending):
                        save()
                        print(f"[{done}/{len(pending)}] Saved {output}")
            except BaseException:
                for future in futures:
                    future.cancel()
                save()
                raise
    finally:
        try:
            # Executor shutdown waits for running calls; retain their results too.
            for future, row in futures.items():
                if future.done() and not future.cancelled():
                    error = future.exception()
                    if error is None or isinstance(error, Exception):
                        collect(future, row)
            save()
        finally:
            client.close()
    statuses = Counter("error" if "error" in row[result_key] else "ok"
                       for row in results.values() if result_key in row)
    print(f"Done: {dict(statuses)}")
    return [results[key] for key in sorted(results)]
