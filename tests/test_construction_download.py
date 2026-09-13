"""Offline regressions for image download fallback, caching, and metadata."""

import json
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from data_construction import images


@pytest.fixture
def record():
    return {"_id": "example", "image_url": "https://example.org/photo-articleLarge.jpg?quality=90"}


@pytest.fixture
def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="JPEG")
    return stream.getvalue()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    get = Mock(side_effect=AssertionError("Unexpected network request"))
    monkeypatch.setattr(requests, "get", get)
    return get


def test_download_falls_back_after_jumbo_request_error(tmp_path, record, no_network, image_bytes):
    content = image_bytes
    no_network.side_effect = [
        requests.Timeout("jumbo timed out"),
        SimpleNamespace(status_code=200, content=content),
    ]

    record_id, filename, status = images.download_one(record, tmp_path, timeout=7)

    assert record_id == record["_id"]
    assert filename == images.url_to_filename(record["image_url"])
    assert status == "ok_fallback"
    assert (tmp_path / filename).read_bytes() == content
    assert no_network.call_args_list[0].args == ("https://example.org/photo-jumbo.jpg?quality=90",)
    assert no_network.call_args_list[1].args == (record["image_url"],)
    assert all(call.kwargs == {"timeout": 7} for call in no_network.call_args_list)


def test_successful_jumbo_is_reused_without_network(tmp_path, record, no_network, image_bytes):
    content = image_bytes
    no_network.side_effect = None
    no_network.return_value = SimpleNamespace(status_code=200, content=content)

    record_id, filename, status = images.download_one(record, tmp_path)
    assert status == "ok_jumbo"
    assert filename.endswith("_jumbo.jpg")
    assert (tmp_path / filename).read_bytes() == content
    assert images.download_one(record, tmp_path) == (record_id, filename, "skipped")
    no_network.assert_called_once()


def test_original_cache_is_reused_before_any_download(tmp_path, record, no_network, image_bytes):
    filename = images.url_to_filename(record["image_url"])
    (tmp_path / filename).write_bytes(image_bytes)

    assert images.download_one(record, tmp_path) == (record["_id"], filename, "skipped")
    no_network.assert_not_called()


def test_failed_download_clears_stale_local_image(tmp_path, record, no_network):
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "nested" / "result.json"
    image_dir = tmp_path / "images"
    record["local_image"] = "stale.jpg"
    metadata.write_text(json.dumps([record]))
    no_network.side_effect = [
        SimpleNamespace(status_code=404, content=b"missing"),
        SimpleNamespace(status_code=200, content=b"too small"),
    ]

    images.download_main([
        "--metadata", str(metadata), "--image_dir", str(image_dir),
        "--output", str(output), "--max_workers", "1",
    ])

    result = json.loads(output.read_text())
    assert result == [{key: value for key, value in record.items() if key != "local_image"}]
    assert list(image_dir.iterdir()) == []
    assert json.loads(metadata.read_text())[0]["local_image"] == "stale.jpg"


def test_zero_limit_downloads_no_records(tmp_path, record, no_network):
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "result.json"
    metadata.write_text(json.dumps([record]))

    images.download_main([
        "--metadata", str(metadata), "--image-dir", str(tmp_path / "images"),
        "--output", str(output), "--max-workers", "1", "--limit", "0",
    ])

    assert json.loads(output.read_text()) == []
    no_network.assert_not_called()


@pytest.mark.parametrize("argument,value", [("--max-workers", "0"), ("--max_workers", "-1"), ("--limit", "-1")])
def test_invalid_counts_rejected_before_reading_metadata(argument, value):
    with pytest.raises(SystemExit) as error:
        images.download_main([argument, value])
    assert error.value.code == 2
