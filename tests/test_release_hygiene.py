"""Regression checks for release metadata and history scans."""

import subprocess

import pytest

from scripts import check_release


def test_history_scan_checks_deleted_credentials_without_printing_values(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.name", "Release test")
    git("config", "user.email", "test@example.invalid")
    token = "sk-" + "a" * 40
    credential = tmp_path / "old.env"
    credential.write_text("OPENAI_API_KEY=" + token + "\n")
    git("add", "old.env")
    git("commit", "-m", "Synthetic credential fixture")
    git("rm", "old.env")
    git("commit", "-m", "Remove fixture")
    monkeypatch.setattr(check_release, "ROOT", tmp_path)
    errors, count = check_release.check_history()
    assert count >= 1
    assert len(errors) == 1 and "old.env" in errors[0]
    assert token not in errors[0]


def test_heading_anchors_handle_unicode_and_repeated_sections():
    anchors = check_release.markdown_anchors(
        "## Construct image–claim pairs\n## Training\n## Training\n## **MIC** overview\n"
    )
    assert anchors == {"construct-imageclaim-pairs", "training", "training-1", "mic-overview"}


@pytest.mark.parametrize("section", [
    "News", "Abstract", "tl;dr", "Datasets", "Environment", "Experiments", "Citation", "Disclaimer",
])
def test_ukp_publication_metadata_is_required(tmp_path, monkeypatch, section):
    readme = (check_release.ROOT / "README.md").read_text()
    notice = (check_release.ROOT / "NOTICE.txt").read_text()
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "NOTICE.txt").write_text(notice)
    monkeypatch.setattr(check_release, "ROOT", tmp_path)
    assert check_release.check_ukp_metadata() == []
    (tmp_path / "README.md").write_text(readme.replace(f"## {section}\n", "## Notes\n"))
    assert any(f"## {section}" in error for error in check_release.check_ukp_metadata())
