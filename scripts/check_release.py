"""Check release source hygiene without loading models or calling services."""

import argparse
import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u4e00-\u9fff]")
PERSONAL_PATH = re.compile(r"/vast/users/|/Users/zengrh|/home/(?:zengrh|preslav)")
SECRET = re.compile(
    r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|hf_[A-Za-z0-9]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def markdown_anchors(text):
    """GitHub-style anchors for ordinary Markdown headings, including duplicates."""
    anchors, seen = set(), {}
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        anchors.add(f"{slug}-{count}" if count else slug)
        seen[slug] = count + 1
    return anchors


def check_history():
    """Scan reachable Git blobs without ever printing credential values."""
    objects = subprocess.check_output(["git", "rev-list", "--objects", "--all"], cwd=ROOT).splitlines()
    errors, count = [], 0
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        for row in objects:
            oid, _, name = row.partition(b" ")
            process.stdin.write(oid + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError("Cannot read a Git object for history scanning")
            data = process.stdout.read(int(header[2]))
            process.stdout.read(1)
            if header[1] != b"blob":
                continue
            count += 1
            if SECRET.search(data.decode("utf-8", errors="replace")):
                path = name.decode("utf-8", errors="replace") or "unnamed blob"
                errors.append(f"Git history {path} ({oid.decode()[:12]}): possible credential (value suppressed)")
        process.stdin.close()
    return errors, count


def check_ukp_metadata():
    """Retain the Misviz publication fields required by this repository owner."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "Contact person:", "[UKP Lab]", "[TU Darmstadt]",
        "This repository contains experimental software and is published for the sole "
        "purpose of giving additional background details on the respective publication.",
    ]
    errors = [f"README.md: missing UKP metadata: {field}" for field in required if field not in readme]
    headings = set(re.findall(r"^## (.+)$", readme, re.M))
    for section in ("News", "Abstract", "tl;dr", "Datasets", "Environment", "Experiments", "Citation", "Disclaimer"):
        if section not in headings:
            errors.append(f"README.md: missing required publication section: ## {section}")
    notice = (ROOT / "NOTICE.txt").read_text(encoding="utf-8") if (ROOT / "NOTICE.txt").is_file() else ""
    if "Ubiquitous Knowledge Processing (UKP) Lab" not in notice:
        errors.append("NOTICE.txt: missing UKP attribution")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also scan reachable Git history for credentials")
    args = parser.parse_args()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    added = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    errors = []
    python_count = 0
    for name in sorted(set(tracked + added) - {""}):
        path = ROOT / name
        if path.is_symlink() and not path.exists():
            errors.append(f"{name}: broken symlink")
        if not path.is_file():
            continue
        if ("__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store"
                or (path.name.startswith(".env") and path.name != ".env.example")):
            errors.append(f"{name}: runtime file included in release")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        # The checker contains its own patterns; do not scan those as findings.
        if name != "scripts/check_release.py":
            if PERSONAL_PATH.search(text):
                errors.append(f"{name}: private machine path")
            if SECRET.search(text):
                errors.append(f"{name}: possible credential (value suppressed)")
        if path.suffix == ".py":
            python_count += 1
            try:
                ast.parse(text, filename=name)
                for token in tokenize.generate_tokens(io.StringIO(text).readline):
                    if token.type == tokenize.COMMENT and HAN.search(token.string):
                        errors.append(f"{name}:{token.start[0]}: Chinese source comment")
            except (SyntaxError, tokenize.TokenError, IndentationError) as exc:
                errors.append(f"{name}: Python syntax error: {exc}")
        elif path.suffix in {".sh", ".yaml", ".yml"}:
            for line_number, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("#") and HAN.search(line):
                    errors.append(f"{name}:{line_number}: Chinese source comment")
        # Validate local documentation links in MIC-authored pages.
        if path.suffix == ".md" and not name.startswith(("LlamaFactory/", "training/verl/")):
            for target in re.findall(r"\]\(([^\s)]+)\)", text):
                if "://" in target or target.startswith("mailto:"):
                    continue
                link, _, fragment = target.partition("#")
                destination = path.parent / link if link else path
                if not destination.exists():
                    errors.append(f"{name}: missing local link {target}")
                elif fragment and destination.suffix == ".md":
                    if fragment not in markdown_anchors(destination.read_text(encoding="utf-8")):
                        errors.append(f"{name}: missing heading anchor {target}")
    for name in ["LICENSE", "NOTICE.txt", "requirements.txt", "THIRD_PARTY_NOTICES.md", "LlamaFactory/LICENSE", "training/verl/LICENSE", "training/verl/Notice.txt"]:
        if not (ROOT / name).is_file():
            errors.append(f"Missing required release file: {name}")
    errors.extend(check_ukp_metadata())
    if args.history:
        history_errors, blobs = check_history()
        errors.extend(history_errors)
        print(f"Scanned {blobs} reachable Git blobs for credential patterns.")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Release checks passed: {python_count} Python files; source comments, paths, local links, notices, and credential patterns checked.")


if __name__ == "__main__":
    main()
