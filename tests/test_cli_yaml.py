import argparse
import pytest

from ai_models import __main__ as cli


def _patch_cli_catalogs(monkeypatch):
    monkeypatch.setattr(cli, "available_inputs", lambda: {"mars": object(), "cds": object(), "file": object()})
    monkeypatch.setattr(cli, "available_outputs", lambda: {"file": object(), "none": object()})
    monkeypatch.setattr(cli, "available_models", lambda: {"fourcastnetv2-small": object()})


def test_main_accepts_yaml_only_configuration(tmp_path, monkeypatch):
    _patch_cli_catalogs(monkeypatch)

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
model: fourcastnetv2-small
run:
  input: cds
  date: 2023-01-10
  time: "00:00"
  lead_time: 48
runtime:
  assets_dir: ./assets/fourcastnetv2-small
cli:
  output: none
targets:
  - name: baseline
    metric: mean-square
""".strip()
    )

    captured = {}

    def fake_run(cfg, model_args):
        captured["cfg"] = cfg
        captured["model_args"] = model_args

    monkeypatch.setattr(cli, "run", fake_run)

    cli._main(["--yaml", str(yaml_path)])

    assert captured["cfg"]["model"] == "fourcastnetv2-small"
    assert captured["cfg"]["input"] == "cds"
    assert captured["cfg"]["date"] == "20230110"
    assert captured["cfg"]["time"] == 0
    assert captured["cfg"]["lead_time"] == 48
    assert captured["cfg"]["output"] == "none"
    assert captured["cfg"]["assets"] == str((tmp_path / "assets/fourcastnetv2-small").resolve())
    assert captured["model_args"] == ["--sensitivity-config", str(yaml_path.resolve())]


def test_main_cli_options_override_yaml_values(tmp_path, monkeypatch):
    _patch_cli_catalogs(monkeypatch)

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
model: fourcastnetv2-small
run:
  input: cds
  date: 2023-01-10
  time: 0000
""".strip()
    )

    captured = {}

    def fake_run(cfg, model_args):
        captured["cfg"] = cfg
        captured["model_args"] = model_args

    monkeypatch.setattr(cli, "run", fake_run)

    cli._main(
        [
            "--yaml",
            str(yaml_path),
            "--input",
            "mars",
            "--date",
            "20240101",
            "--time",
            "1200",
        ]
    )

    assert captured["cfg"]["input"] == "mars"
    assert captured["cfg"]["date"] == "20240101"
    assert captured["cfg"]["time"] == 1200


def test_main_maps_additional_standard_cli_options_from_yaml(tmp_path, monkeypatch):
    _patch_cli_catalogs(monkeypatch)

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
model: fourcastnetv2-small
run:
  input: cds
  output: none
  date: 20230110
  time: 0000
  lead_time: 24
  only_gpu: true
runtime:
  assets_dir: ./assets/fourcastnetv2-small
cli:
  debug: true
  verbose: 2
  json: true
  retrieve_fields_type: prognostics
  retrieve_only_one_date: true
  num_threads: 4
  model_version: latest
  metadata:
    expver: test
    class: rd
""".strip()
    )

    captured = {}

    def fake_run(cfg, model_args):
        captured["cfg"] = cfg
        captured["model_args"] = model_args

    monkeypatch.setattr(cli, "run", fake_run)

    cli._main(["--yaml", str(yaml_path)])

    assert captured["cfg"]["debug"] is True
    assert captured["cfg"]["verbose"] == 2
    assert captured["cfg"]["json"] is True
    assert captured["cfg"]["retrieve_fields_type"] == "prognostics"
    assert captured["cfg"]["retrieve_only_one_date"] is True
    assert captured["cfg"]["num_threads"] == 4
    assert captured["cfg"]["metadata"] == {"expver": "test", "class": "rd"}


def test_main_requires_model_from_cli_or_yaml(tmp_path, monkeypatch):
    _patch_cli_catalogs(monkeypatch)

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
run:
  input: cds
""".strip()
    )

    with pytest.raises(SystemExit):
        cli._main(["--yaml", str(yaml_path)])


def test_as_date_rejects_short_positive_integer():
    with pytest.raises(ValueError):
        cli._as_date(12)


def test_as_date_accepts_zero_relative_day():
    assert cli._as_date(0) == "0"
    assert cli._as_date("0") == "0"


def test_yaml_model_args_must_be_list(tmp_path):
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
model: fourcastnetv2-small
cli:
  model_args: "--plot-coastlines"
""".strip()
    )

    parser = argparse.ArgumentParser()
    args = type(
        "Args",
        (),
        {
            "yaml": str(yaml_path),
            "model": None,
            "input": "mars",
            "output": "file",
            "date": "-1",
            "time": 12,
            "assets": ".",
            "path": None,
            "file": None,
            "download_assets": False,
            "only_gpu": False,
            "deterministic": False,
            "num_threads": 1,
            "model_version": "latest",
            "remote_execution": False,
            "assets_sub_directory": False,
        },
    )()

    with pytest.raises(SystemExit):
        cli._apply_yaml_overrides(parser, args, [], ["--yaml", str(yaml_path)])


def test_main_accepts_model_from_run_section(tmp_path, monkeypatch):
    _patch_cli_catalogs(monkeypatch)

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        """
run:
  model: fourcastnetv2-small
  input: cds
  date: 20230110
  time: 0000
""".strip()
    )

    captured = {}

    def fake_run(cfg, model_args):
        captured["cfg"] = cfg
        captured["model_args"] = model_args

    monkeypatch.setattr(cli, "run", fake_run)

    cli._main(["--yaml", str(yaml_path)])

    assert captured["cfg"]["model"] == "fourcastnetv2-small"
