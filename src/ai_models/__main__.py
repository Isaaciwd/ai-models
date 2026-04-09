# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import datetime
import logging
import os
import shlex
import sys
from pathlib import Path

import earthkit.data as ekd
import yaml

from .inputs import available_inputs
from .model import Timer
from .model import available_models
from .model import load_model
from .outputs import available_outputs

ekd.settings.set("cache-policy", "user")

LOG = logging.getLogger(__name__)


def _provided_options(argv):
    options = set()
    for token in argv:
        if token.startswith("--"):
            options.add(token.split("=", 1)[0])
    return options


def _unknown_has_option(unknownargs, option):
    for token in unknownargs:
        if token == option or token.startswith(f"{option}="):
            return True
    return False


def _mapping(config, key):
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _first_value(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _as_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value")


def _as_int(value, name):
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def _as_choice(value, name, allowed):
    text = str(value)
    if text not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {options}")
    return text


def _as_metadata_list(value):
    if isinstance(value, dict):
        return [f"{key}={item}" for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            text = str(item)
            if "=" not in text:
                raise ValueError("metadata entries must be KEY=VALUE")
            result.append(text)
        return result
    text = str(value)
    if "=" not in text:
        raise ValueError("metadata entries must be KEY=VALUE")
    return [text]


def _as_date(value):
    if isinstance(value, int):
        if value < 0:
            return str(value)
        if value == 0:
            return "0"
        text = str(value)
        if len(text) != 8:
            raise ValueError("date must be YYYYMMDD, YYYY-MM-DD, or a relative integer day")
        try:
            datetime.datetime.strptime(text, "%Y%m%d")
        except ValueError as error:
            raise ValueError("date must be a valid calendar date in YYYYMMDD format") from error
        return text

    text = str(value).strip()
    if text.startswith("-") and text[1:].isdigit():
        return text

    if "-" in text:
        try:
            parsed = datetime.datetime.strptime(text, "%Y-%m-%d")
        except ValueError as error:
            raise ValueError("date must be YYYYMMDD, YYYY-MM-DD, or a relative integer day") from error
        return parsed.strftime("%Y%m%d")

    if text.isdigit():
        if text == "0":
            return text
        if len(text) != 8:
            raise ValueError("date must be YYYYMMDD, YYYY-MM-DD, or a relative integer day")
        if len(text) == 8:
            try:
                datetime.datetime.strptime(text, "%Y%m%d")
            except ValueError as error:
                raise ValueError("date must be a valid calendar date in YYYYMMDD format") from error
        return text

    raise ValueError("date must be YYYYMMDD, YYYY-MM-DD, or a relative integer day")


def _as_time(value):
    if isinstance(value, int):
        numeric = value
    else:
        text = str(value).strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError("time must be HHMM or HH:MM")
            numeric = int(parts[0]) * 100 + int(parts[1])
        else:
            if not text.isdigit():
                raise ValueError("time must be HHMM or HH:MM")
            numeric = int(text)

    if numeric < 0:
        raise ValueError("time must be non-negative")

    if numeric < 100:
        hour = numeric
        minute = 0
    else:
        hour = numeric // 100
        minute = numeric % 100

    if hour > 23 or minute > 59:
        raise ValueError("time must be a valid clock time in HHMM or HH:MM")

    return numeric


def _format_yaml_string(value, yaml_path):
    text = str(value)
    context = {
        "repo_root": str(yaml_path.parent),
        "yaml_dir": str(yaml_path.parent),
        "cwd": os.getcwd(),
    }
    if "{" in text and "}" in text:
        try:
            text = text.format(**context)
        except KeyError as error:
            raise ValueError(
                f"Unknown placeholder '{{{error.args[0]}}}' in YAML value; supported placeholders are '{{repo_root}}', '{{yaml_dir}}', and '{{cwd}}'"
            ) from error
    return text


def _resolve_yaml_path(value, yaml_path):
    rendered = _format_yaml_string(value, yaml_path)
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = (yaml_path.parent / path).resolve()
    return str(path)


def _apply_yaml_overrides(parser, args, unknownargs, argv):
    if not args.yaml:
        return args, unknownargs

    yaml_path = Path(args.yaml).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = (Path.cwd() / yaml_path).resolve()

    if not yaml_path.exists():
        parser.error(f"YAML file not found: {yaml_path}")

    try:
        config = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception as error:
        parser.error(f"Failed to read YAML file {yaml_path}: {error}")

    if not isinstance(config, dict):
        parser.error("YAML root must be a mapping")

    args.yaml = str(yaml_path)
    provided = _provided_options(argv)

    try:
        cli_cfg = _mapping(config, "cli")
        run_cfg = _mapping(config, "run")
        runtime_cfg = _mapping(config, "runtime")
    except ValueError as error:
        parser.error(str(error))

    def set_option(dest, value, option_names, transform=None):
        if value is None:
            return
        if any(name in provided for name in option_names):
            return
        try:
            setattr(args, dest, transform(value) if transform else value)
        except ValueError as error:
            parser.error(str(error))

    if args.model is None:
        model_value = _first_value(cli_cfg.get("model"), run_cfg.get("model"), config.get("model"))
        if model_value is not None:
            args.model = str(model_value)

    set_option(
        "input",
        _first_value(cli_cfg.get("input"), run_cfg.get("input"), config.get("input")),
        ["--input"],
        lambda value: _as_choice(value, "input", available_inputs().keys()),
    )

    output_value = _first_value(cli_cfg.get("output"), run_cfg.get("output"))
    if output_value is None:
        raw_output = config.get("output")
        if isinstance(raw_output, str):
            output_value = raw_output
    set_option("output", output_value, ["--output"], lambda value: _as_choice(value, "output", available_outputs().keys()))

    set_option(
        "debug",
        _first_value(cli_cfg.get("debug"), config.get("debug")),
        ["--debug"],
        lambda value: _as_bool(value, "debug"),
    )

    set_option(
        "verbose",
        _first_value(cli_cfg.get("verbose"), config.get("verbose")),
        ["-v", "--verbose"],
        lambda value: _as_int(value, "verbose"),
    )

    set_option(
        "retrieve_requests",
        _first_value(cli_cfg.get("retrieve_requests"), config.get("retrieve_requests")),
        ["--retrieve-requests"],
        lambda value: _as_bool(value, "retrieve_requests"),
    )

    set_option(
        "archive_requests",
        _first_value(cli_cfg.get("archive_requests"), config.get("archive_requests")),
        ["--archive-requests"],
        lambda value: _resolve_yaml_path(value, yaml_path),
    )

    set_option(
        "requests_extra",
        _first_value(cli_cfg.get("requests_extra"), config.get("requests_extra")),
        ["--requests-extra"],
        lambda value: str(value),
    )

    set_option(
        "json",
        _first_value(cli_cfg.get("json"), config.get("json")),
        ["--json"],
        lambda value: _as_bool(value, "json"),
    )

    set_option(
        "retrieve_fields_type",
        _first_value(cli_cfg.get("retrieve_fields_type"), config.get("retrieve_fields_type")),
        ["--retrieve-fields-type"],
        lambda value: _as_choice(value, "retrieve_fields_type", {"constants", "prognostics", "all"}),
    )

    set_option(
        "retrieve_only_one_date",
        _first_value(cli_cfg.get("retrieve_only_one_date"), config.get("retrieve_only_one_date")),
        ["--retrieve-only-one-date"],
        lambda value: _as_bool(value, "retrieve_only_one_date"),
    )

    set_option(
        "dump_provenance",
        _first_value(cli_cfg.get("dump_provenance"), config.get("dump_provenance")),
        ["--dump-provenance"],
        lambda value: _resolve_yaml_path(value, yaml_path),
    )

    set_option(
        "date",
        _first_value(cli_cfg.get("date"), run_cfg.get("date"), config.get("date")),
        ["--date"],
        _as_date,
    )

    set_option(
        "time",
        _first_value(cli_cfg.get("time"), run_cfg.get("time"), config.get("time")),
        ["--time"],
        _as_time,
    )

    set_option(
        "lead_time",
        _first_value(cli_cfg.get("lead_time"), run_cfg.get("lead_time"), config.get("lead_time")),
        ["--lead-time"],
        lambda value: _as_int(value, "lead_time"),
    )

    set_option(
        "assets",
        _first_value(cli_cfg.get("assets"), runtime_cfg.get("assets_dir"), config.get("assets")),
        ["--assets"],
        lambda value: _resolve_yaml_path(value, yaml_path),
    )

    set_option(
        "path",
        _first_value(cli_cfg.get("path"), config.get("path")),
        ["--path"],
        lambda value: _resolve_yaml_path(value, yaml_path),
    )

    set_option(
        "file",
        _first_value(cli_cfg.get("file"), config.get("file")),
        ["--file"],
        lambda value: _resolve_yaml_path(value, yaml_path),
    )

    set_option(
        "fields",
        _first_value(cli_cfg.get("fields"), config.get("fields")),
        ["--fields"],
        lambda value: _as_bool(value, "fields"),
    )

    set_option(
        "expver",
        _first_value(cli_cfg.get("expver"), config.get("expver")),
        ["--expver"],
        lambda value: str(value),
    )

    set_option(
        "class_",
        _first_value(cli_cfg.get("class"), cli_cfg.get("class_"), config.get("class"), config.get("class_")),
        ["--class"],
        lambda value: str(value),
    )

    metadata = _first_value(cli_cfg.get("metadata"), config.get("metadata"))
    if metadata is not None and "--metadata" not in provided:
        try:
            args.metadata = _as_metadata_list(metadata)
        except ValueError as error:
            parser.error(str(error))

    set_option(
        "download_assets",
        _first_value(cli_cfg.get("download_assets"), config.get("download_assets")),
        ["--download-assets"],
        lambda value: _as_bool(value, "download_assets"),
    )

    set_option(
        "only_gpu",
        _first_value(cli_cfg.get("only_gpu"), run_cfg.get("only_gpu"), config.get("only_gpu")),
        ["--only-gpu"],
        lambda value: _as_bool(value, "only_gpu"),
    )

    set_option(
        "deterministic",
        _first_value(cli_cfg.get("deterministic"), config.get("deterministic")),
        ["--deterministic"],
        lambda value: _as_bool(value, "deterministic"),
    )

    set_option(
        "num_threads",
        _first_value(cli_cfg.get("num_threads"), config.get("num_threads")),
        ["--num-threads"],
        lambda value: _as_int(value, "num_threads"),
    )

    set_option(
        "model_version",
        _first_value(cli_cfg.get("model_version"), config.get("model_version")),
        ["--model-version"],
        lambda value: str(value),
    )

    set_option(
        "hindcast_reference_year",
        _first_value(cli_cfg.get("hindcast_reference_year"), config.get("hindcast_reference_year")),
        ["--hindcast-reference-year"],
        lambda value: str(value),
    )

    set_option(
        "hindcast_reference_date",
        _first_value(cli_cfg.get("hindcast_reference_date"), config.get("hindcast_reference_date")),
        ["--hindcast-reference-date"],
        lambda value: str(value),
    )

    set_option(
        "staging_dates",
        _first_value(cli_cfg.get("staging_dates"), config.get("staging_dates")),
        ["--staging-dates"],
        lambda value: _resolve_yaml_path(value, yaml_path),
    )

    set_option(
        "remote_execution",
        _first_value(
            cli_cfg.get("remote_execution"),
            cli_cfg.get("remote"),
            config.get("remote_execution"),
            config.get("remote"),
        ),
        ["--remote"],
        lambda value: _as_bool(value, "remote_execution"),
    )

    set_option(
        "assets_sub_directory",
        _first_value(cli_cfg.get("assets_sub_directory"), config.get("assets_sub_directory")),
        ["--assets-sub-directory", "--no-assets-sub-directory"],
        lambda value: _as_bool(value, "assets_sub_directory"),
    )

    model_args = _first_value(cli_cfg.get("model_args"), config.get("model_args"))
    if model_args is not None:
        if not isinstance(model_args, list):
            parser.error("model_args must be a list when provided in YAML")
        unknownargs.extend(str(item) for item in model_args)

    sensitivity_config = _first_value(
        cli_cfg.get("sensitivity_config"),
        run_cfg.get("sensitivity_config"),
        config.get("sensitivity_config"),
    )

    has_sensitivity_sections = (
        "targets" in config
        or "plotting" in config
        or isinstance(config.get("output"), dict)
    )
    if sensitivity_config is None and has_sensitivity_sections:
        sensitivity_config = str(yaml_path)

    if sensitivity_config is not None and not _unknown_has_option(unknownargs, "--sensitivity-config"):
        if str(sensitivity_config) == str(yaml_path):
            resolved = str(yaml_path)
        else:
            try:
                resolved = _resolve_yaml_path(sensitivity_config, yaml_path)
            except ValueError as error:
                parser.error(str(error))
        unknownargs.extend(["--sensitivity-config", resolved])

    return args, unknownargs


def _main(argv):
    parser = argparse.ArgumentParser()

    # See https://github.com/pytorch/pytorch/issues/77764
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    parser.add_argument(
        "--models",
        action="store_true",
        help="List models and exit",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn on debug",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity",
    )

    parser.add_argument(
        "--yaml",
        metavar="FILE",
        help=(
            "YAML run configuration file. "
            "Supports mapping model/input/date/time/lead_time and related CLI options to a single file."
        ),
    )

    parser.add_argument(
        "--retrieve-requests",
        help=("Print mars requests to stdout." "Use --requests-extra to extend or overide the requests. "),
        action="store_true",
    )

    parser.add_argument(
        "--archive-requests",
        help=("Save mars archive requests to FILE." "Use --requests-extra to extend or overide the requests. "),
        metavar="FILE",
    )

    parser.add_argument(
        "--requests-extra",
        help=("Extends the retrieve or archive requests with a list of key1=value1,key2=value."),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=("Dump the requests in JSON format."),
    )

    parser.add_argument(
        "--retrieve-fields-type",
        help="Type of field to retrieve. To use with --retrieve-requests.",
        choices=["constants", "prognostics", "all"],
        default="all",
    )

    parser.add_argument(
        "--retrieve-only-one-date",
        help="Only retrieve the last date/time. To use with --retrieve-requests.",
        action="store_true",
    )

    parser.add_argument(
        "--dump-provenance",
        metavar="FILE",
        help=("Dump information for tracking provenance."),
    )

    parser.add_argument(
        "--input",
        default="mars",
        help="Source to use",
        choices=sorted(available_inputs()),
    )

    parser.add_argument(
        "--file",
        help="Source to use if source=file",
    )

    parser.add_argument(
        "--output",
        default="file",
        help="Where to output the results",
        choices=sorted(available_outputs()),
    )

    parser.add_argument(
        "--date",
        default="-1",
        help="For which analysis date to start the inference (default: yesterday)",
    )

    parser.add_argument(
        "--time",
        type=int,
        default=12,
        help="For which analysis time to start the inference (default: 12)",
    )

    parser.add_argument(
        "--assets",
        default=os.environ.get("AI_MODELS_ASSETS", "."),
        help="Path to directory containing the weights and other assets",
    )

    parser.add_argument(
        "--assets-sub-directory",
        help="Load assets from a subdirectory of --assets based on the name of the model.",
        action=argparse.BooleanOptionalAction,
    )

    parser.parse_args(["--no-assets-sub-directory"])

    parser.add_argument(
        "--assets-list",
        help="List the assets used by the model",
        action="store_true",
    )

    parser.add_argument(
        "--download-assets",
        help="Download assets if they do not exists.",
        action="store_true",
    )

    parser.add_argument(
        "--path",
        help="Path where to write the output of the model",
    )

    parser.add_argument(
        "--fields",
        help="Show the fields needed as input for the model",
        action="store_true",
    )

    parser.add_argument(
        "--expver",
        help="Set the experiment version of the model output. Has higher priority than --metadata.",
    )

    parser.add_argument(
        "--class",
        help="Set the 'class' metadata of the model output. Has higher priority than --metadata.",
        metavar="CLASS",
        dest="class_",
    )

    parser.add_argument(
        "--metadata",
        help="Set additional metadata metadata in the model output",
        metavar="KEY=VALUE",
        action="append",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads. Only relevant for some models.",
    )

    parser.add_argument(
        "--lead-time",
        type=int,
        default=240,
        help="Length of forecast in hours.",
    )

    parser.add_argument(
        "--hindcast-reference-year",
        help="For encoding hincast-like outputs",
    )

    parser.add_argument(
        "--hindcast-reference-date",
        help="For encoding hincast-like outputs",
    )

    parser.add_argument(
        "--staging-dates",
        help="For encoding hincast-like outputs",
    )

    parser.add_argument(
        "--only-gpu",
        help="Fail if GPU is not available",
        action="store_true",
    )

    parser.add_argument(
        "--deterministic",
        help="Fail if GPU is not available",
        action="store_true",
    )

    # TODO: deprecate that option
    parser.add_argument(
        "--model-version",
        default="latest",
        help="Model version",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print ai-models version and exit",
    )

    parser.add_argument(
        "model",
        metavar="MODEL",
        nargs="?",
        choices=available_models() if "--remote" not in argv else None,
        help="The model to run",
    )

    parser.add_argument(
        "--remote",
        help="Enable remote execution, read url and token from ~/.config/ai-models/api.yaml",
        action="store_true",
        dest="remote_execution",
        default=(os.environ.get("AI_MODELS_REMOTE", "0") == "1"),
    )

    args, unknownargs = parser.parse_known_args(argv)

    if args.version:
        from ai_models import __version__

        print(__version__)
        sys.exit(0)

    del args.version

    if args.models:
        if args.remote_execution:
            from .remote import RemoteAPI

            api = RemoteAPI()
            models = api.models()
            if len(models) == 0:
                print(f"No remote models available on {api.url}")
                sys.exit(0)
            print(f"Models available on remote server {api.url}")
        else:
            models = available_models()

        for p in sorted(models):
            print(p)
        sys.exit(0)

    args, unknownargs = _apply_yaml_overrides(parser, args, unknownargs, argv)

    if args.model is None:
        parser.error("You need to specify MODEL, either as a positional argument or in the YAML file")

    if not args.remote_execution and args.model not in available_models():
        parser.error(f"Unknown model '{args.model}'. Use --models to list available models")

    if args.assets_sub_directory:
        args.assets = os.path.join(args.assets, args.model)

    if args.path is None:
        args.path = f"{args.model}.grib"

    if args.file is not None:
        args.input = "file"

    if not args.fields and not args.retrieve_requests:
        logging.basicConfig(
            level="DEBUG" if args.debug else "INFO",
            format="%(asctime)s %(levelname)s %(message)s",
        )

    if args.metadata is None:
        args.metadata = []

    args.metadata = dict(kv.split("=") for kv in args.metadata)

    if args.expver is not None:
        args.metadata["expver"] = args.expver

    if args.class_ is not None:
        args.metadata["class"] = args.class_

    if args.requests_extra:
        if not args.retrieve_requests and not args.archive_requests:
            parser.error("You need to specify --retrieve-requests or --archive-requests")

    run(vars(args), unknownargs)


def run(cfg: dict, model_args: list):
    if cfg["remote_execution"]:
        from .remote import RemoteModel

        model = RemoteModel(**cfg, model_args=model_args)
    else:
        model = load_model(cfg["model"], **cfg, model_args=model_args)

    if cfg["fields"]:
        model.print_fields()
        sys.exit(0)

    # This logic is a bit convoluted, but it is for backwards compatibility.
    if cfg["retrieve_requests"] or (cfg["requests_extra"] and not cfg["archive_requests"]):
        model.print_requests()
        sys.exit(0)

    if cfg["assets_list"]:
        model.print_assets_list()
        sys.exit(0)

    try:
        model.run()
    except FileNotFoundError as e:
        LOG.exception(e)
        LOG.error(
            "It is possible that some files required by %s are missing.",
            cfg["model"],
        )
        LOG.error("Rerun the command as:")
        LOG.error(
            "   %s",
            shlex.join([sys.argv[0], "--download-assets"] + sys.argv[1:]),
        )
        sys.exit(1)

    model.finalise()

    if cfg["dump_provenance"]:
        with Timer("Collect provenance information"):
            with open(cfg["dump_provenance"], "w") as f:
                prov = model.provenance()
                import json  # import here so it is not listed in provenance

                json.dump(prov, f, indent=4)


def main():
    with Timer("Total time"):
        _main(sys.argv[1:])


if __name__ == "__main__":
    main()
