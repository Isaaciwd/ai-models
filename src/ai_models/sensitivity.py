# (C) Copyright 2026 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import datetime
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import yaml

LOG = logging.getLogger(__name__)


@dataclass
class SensitivityTarget:
    name: str
    field: Optional[str]
    area: Optional[Tuple[float, float, float, float]]
    metric: str


def parse_target_area(value):
    if value is None:
        return None

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        raise ValueError("Target area must be a comma-delimited string or a 4-item list")

    if len(parts) != 4:
        raise ValueError("Target area must be 'north,west,south,east'")

    north, west, south, east = (float(part) for part in parts)
    if north < south:
        raise ValueError("Target area must satisfy north >= south")

    return (north, west % 360.0, south, east % 360.0)


def target_slug(value, fallback):
    if value is None:
        value = fallback
    value = str(value).strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or fallback


def add_sensitivity_parser_arguments(parser):
    parser.add_argument(
        "--sensitivity-config",
        help="YAML file defining sensitivity targets, output, plotting, and optional run settings.",
    )
    parser.add_argument(
        "--sensitivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Backpropagate a scalar objective from the final forecast state to the input state.",
    )
    parser.add_argument(
        "--sensitivity-metric",
        choices=("mean", "mean-square"),
        default="mean-square",
        help="Default scalar objective for configured targets.",
    )
    parser.add_argument(
        "--sensitivity-path",
        help="Path to the NetCDF file written for input sensitivities.",
    )
    parser.add_argument(
        "--summary-path",
        help="Optional JSON summary path. Defaults to sensitivity path with .json suffix.",
    )
    parser.add_argument(
        "--target-field",
        help="Final forecast field to target, using native names like r850 or 2t.",
    )
    parser.add_argument(
        "--target-param",
        help="Forecast parameter to target, such as r, t, u, v, z, 2t, or tcwv.",
    )
    parser.add_argument(
        "--target-level",
        type=int,
        help="Pressure level for --target-param when targeting pressure-level fields.",
    )
    parser.add_argument(
        "--target-area",
        help="Lat/lon selection as north,west,south,east in degrees on the model grid.",
    )
    parser.add_argument(
        "--plot-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write PNG plots for the saved sensitivity outputs. Defaults to on in sensitivity mode.",
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=6,
        help="Number of top input channels to include in the summary plot.",
    )
    parser.add_argument(
        "--plot-area",
        help="Lat/lon plotting region as north,west,south,east in degrees.",
    )
    parser.add_argument(
        "--plot-prefix",
        help="Prefix for generated sensitivity plot filenames.",
    )
    parser.add_argument(
        "--plot-coastlines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw continent coastlines in sensitivity plots when cartopy is available.",
    )
    parser.add_argument(
        "--plot-signed-gradients",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write signed-gradient plots with a diverging color map.",
    )


def _channel_maps(gradient):
    if gradient.ndim == 4:
        return gradient[0]
    return gradient


def _channel_sensitivity_scores(gradient_maps):
    return np.abs(gradient_maps).mean(axis=(-2, -1))


def _total_sensitivity_map(gradient_maps):
    return np.abs(gradient_maps).mean(axis=0)


def _signed_total_sensitivity_map(gradient_maps):
    return gradient_maps.mean(axis=0)


def signed_total_sensitivity_map(gradient_maps):
    return _signed_total_sensitivity_map(gradient_maps)


class SensitivityManager:
    def __init__(self, owner, model_name, default_sensitivity_path):
        self.owner = owner
        self.model_name = model_name
        self.default_sensitivity_path = default_sensitivity_path
        self.targets: List[SensitivityTarget] = []
        self.current_target: Optional[SensitivityTarget] = None
        self.plot_area_bounds = None
        self.config = None
        self.config_path = None
        self._warned_missing_cartopy = False

    def configure(self):
        if not hasattr(self.owner, "plot_signed_gradients"):
            self.owner.plot_signed_gradients = False

        if self.owner.model_checkpointing is None:
            self.owner.model_checkpointing = self.owner.sensitivity

        if self.owner.rollout_checkpointing is None:
            self.owner.rollout_checkpointing = self.owner.sensitivity

        if self.owner.plot_sensitivity is None:
            self.owner.plot_sensitivity = self.owner.sensitivity

        if self.owner.plot_top_k < 0:
            raise ValueError("--plot-top-k must be non-negative")

        if self.owner.sensitivity_path is None:
            self.owner.sensitivity_path = self.default_sensitivity_path

        if self.owner.summary_path is None:
            self.owner.summary_path = os.path.splitext(self.owner.sensitivity_path)[0] + ".json"

        if self.owner.plot_prefix is None:
            self.owner.plot_prefix = os.path.splitext(self.owner.sensitivity_path)[0]

        self.plot_area_bounds = parse_target_area(self.owner.plot_area)
        self.targets = [self.owner.default_target()]
        self.current_target = self.targets[0]

        if self.owner.sensitivity_config:
            self.load_config(self.owner.sensitivity_config)

    def load_config(self, path):
        with open(path) as file_handle:
            config = yaml.safe_load(file_handle) or {}

        if not isinstance(config, dict):
            raise ValueError("Sensitivity config root must be a mapping")

        run_cfg = config.get("run", {})
        if run_cfg:
            lead_time = run_cfg.get("lead_time")
            if lead_time is not None:
                self.owner.lead_time = int(lead_time)
            model_checkpointing = run_cfg.get("model_checkpointing")
            if model_checkpointing is not None:
                self.owner.model_checkpointing = bool(model_checkpointing)
            rollout_checkpointing = run_cfg.get("rollout_checkpointing")
            if rollout_checkpointing is not None:
                self.owner.rollout_checkpointing = bool(rollout_checkpointing)

        output_cfg = config.get("output", {})
        if output_cfg:
            self.owner.sensitivity_path = output_cfg.get("path", self.owner.sensitivity_path)
            self.owner.summary_path = output_cfg.get(
                "summary_path", os.path.splitext(self.owner.sensitivity_path)[0] + ".json"
            )

        plotting_cfg = config.get("plotting", {})
        if plotting_cfg:
            if "enabled" in plotting_cfg:
                self.owner.plot_sensitivity = bool(plotting_cfg["enabled"])
            if "top_k" in plotting_cfg:
                self.owner.plot_top_k = int(plotting_cfg["top_k"])
            if "prefix" in plotting_cfg:
                self.owner.plot_prefix = plotting_cfg["prefix"]
            if "area" in plotting_cfg:
                self.plot_area_bounds = parse_target_area(plotting_cfg["area"])
            if "coastlines" in plotting_cfg:
                self.owner.plot_coastlines = bool(plotting_cfg["coastlines"])
            if "signed_gradients" in plotting_cfg:
                self.owner.plot_signed_gradients = bool(plotting_cfg["signed_gradients"])
            elif "signed" in plotting_cfg:
                self.owner.plot_signed_gradients = bool(plotting_cfg["signed"])

        if self.owner.plot_top_k < 0:
            raise ValueError("plotting.top_k must be non-negative")

        if self.owner.plot_prefix is None:
            self.owner.plot_prefix = os.path.splitext(self.owner.sensitivity_path)[0]

        self.targets = self.owner.config_targets(config)
        if self.plot_area_bounds is None and self.targets and self.targets[0] is not None:
            self.plot_area_bounds = self.targets[0].area

        self.current_target = self.targets[0]
        self.config = config
        self.config_path = path
        self.owner.sensitivity = True

    @property
    def target_summary(self):
        if not self.targets:
            return None

        entries = []
        for target in self.targets:
            entry = {
                "name": target.name,
                "field": target.field,
                "metric": target.metric,
            }
            if target.area is not None:
                north, west, south, east = target.area
                entry["area"] = {
                    "north": north,
                    "west": west,
                    "south": south,
                    "east": east,
                }
            entries.append(entry)

        if len(entries) == 1:
            return entries[0]
        return entries

    def objective(self, output, target):
        return self.owner.sensitivity_objective(output, target)

    def total_sensitivity_peak(self, total_map):
        peak_index = np.unravel_index(np.argmax(total_map), total_map.shape)
        lat_index, lon_index = peak_index
        return {
            "latitude": float(self.owner.latitudes[lat_index]),
            "longitude": float(self.owner.longitudes[lon_index]),
            "value": float(total_map[lat_index, lon_index]),
        }

    def target_area_bounds(self):
        if self.current_target is None:
            return None
        return self.current_target.area

    def add_target_area_patch(self, axes, data_crs=None, edgecolor="white"):
        area = self.target_area_bounds()
        if area is None:
            area = self.plot_area_bounds
        if area is None:
            return

        import matplotlib.patches as patches
        import matplotlib.patheffects as patheffects

        north, west, south, east = area
        spans = [(west, east)] if west <= east else [(west, 360.0), (0.0, east)]
        for span_west, span_east in spans:
            patch_kwargs = {}
            if data_crs is not None:
                patch_kwargs["transform"] = data_crs

            halo = "black" if edgecolor == "white" else "white"

            axes.add_patch(
                patches.Rectangle(
                    (span_west, south),
                    span_east - span_west,
                    north - south,
                    fill=False,
                    edgecolor=edgecolor,
                    linewidth=2.2,
                    linestyle="--",
                    zorder=50,
                    clip_on=False,
                    path_effects=[
                        patheffects.Stroke(linewidth=3.4, foreground=halo),
                        patheffects.Normal(),
                    ],
                    **patch_kwargs,
                )
            )

    def maybe_add_coastlines(self, axes, color="white"):
        if not self.owner.plot_coastlines:
            return

        try:
            import cartopy.feature as cfeature

            axes.coastlines(color=color, linewidth=0.6)
            axes.add_feature(cfeature.BORDERS, edgecolor=color, linewidth=0.3)
        except Exception:
            if not self._warned_missing_cartopy:
                LOG.warning("Cartopy is not available; plotting without coastline overlays")
                self._warned_missing_cartopy = True

    @staticmethod
    def symmetric_limits(values):
        vmax = float(np.nanmax(np.abs(values)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-12
        return (-vmax, vmax)

    def apply_plot_limits(self, axes, data_crs=None):
        area = self.plot_area_bounds
        if area is None:
            return

        north, west, south, east = area
        if data_crs is not None and hasattr(axes, "set_extent"):
            if west <= east:
                axes.set_extent([west, east, south, north], crs=data_crs)
            else:
                axes.set_extent([0, 360, south, north], crs=data_crs)
            return

        if west <= east:
            axes.set_xlim(west, east)
        else:
            axes.set_xlim(0, 360)
        axes.set_ylim(south, north)

    @staticmethod
    def _coerce_datetime(date_value, time_value):
        try:
            date = int(date_value)
        except (TypeError, ValueError):
            return None

        if date <= 0:
            base = datetime.datetime.utcnow() + datetime.timedelta(days=date)
            date = base.year * 10000 + base.month * 100 + base.day

        try:
            time = int(time_value)
        except (TypeError, ValueError):
            return None

        if time < 100:
            time *= 100

        try:
            return datetime.datetime(
                date // 10000,
                date % 10000 // 100,
                date % 100,
                time // 100,
                time % 100,
            )
        except ValueError:
            return None

    def target_backpropagation_datetime(self):
        try:
            lead_hours = int(self.owner.lead_time)
        except (TypeError, ValueError):
            return None

        try:
            start = self.owner.start_datetime
        except Exception:
            start = None

        if isinstance(start, datetime.datetime):
            return start + datetime.timedelta(hours=lead_hours)

        base = self._coerce_datetime(
            getattr(self.owner, "date", None),
            getattr(self.owner, "time", None),
        )
        if base is None:
            return None

        return base + datetime.timedelta(hours=lead_hours)

    def plot_context_label(self):
        parts = []

        try:
            parts.append(f"Lead time: +{int(self.owner.lead_time)}h")
        except (TypeError, ValueError):
            pass

        backpropagation_datetime = self.target_backpropagation_datetime()
        if backpropagation_datetime is not None:
            parts.append(f"Backprop from: {backpropagation_datetime:%Y-%m-%d %H:%M} UTC")

        return " | ".join(parts)

    def write_sensitivity_plots(self, gradient_maps, top_channels):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        projection = None
        data_crs = None
        if self.owner.plot_coastlines:
            try:
                import cartopy.crs as ccrs

                projection = ccrs.PlateCarree(central_longitude=180)
                data_crs = ccrs.PlateCarree()
            except Exception:
                if not self._warned_missing_cartopy:
                    LOG.warning("Cartopy is not available; plotting without coastline overlays")
                    self._warned_missing_cartopy = True

        total_map = _total_sensitivity_map(gradient_maps)
        total_path = f"{self.owner.plot_prefix}-total.png"
        paths = []
        context_label = self.plot_context_label()
        extent = (
            0.0,
            360.0,
            float(self.owner.latitudes[-1]),
            float(self.owner.latitudes[0]),
        )

        if projection is not None:
            figure, axes = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})
            image = axes.imshow(
                total_map,
                origin="upper",
                extent=extent,
                cmap="magma",
                transform=data_crs,
            )
        else:
            figure, axes = plt.subplots(figsize=(12, 5))
            image = axes.imshow(total_map, origin="upper", extent=extent, cmap="magma")

        if context_label:
            axes.set_title(f"Total input sensitivity\n{context_label}")
        else:
            axes.set_title("Total input sensitivity")
        axes.set_xlabel("Longitude")
        axes.set_ylabel("Latitude")
        self.maybe_add_coastlines(axes, color="white")
        self.add_target_area_patch(axes, data_crs=data_crs, edgecolor="white")
        self.apply_plot_limits(axes, data_crs=data_crs)
        figure.colorbar(image, ax=axes, shrink=0.8, label="Mean absolute gradient")
        figure.tight_layout()
        figure.savefig(total_path, dpi=150)
        plt.close(figure)
        paths.append(total_path)

        if self.owner.plot_signed_gradients:
            signed_total_map = _signed_total_sensitivity_map(gradient_maps)
            signed_total_path = f"{self.owner.plot_prefix}-total-signed.png"
            vmin, vmax = self.symmetric_limits(signed_total_map)

            if projection is not None:
                figure, axes = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})
                image = axes.imshow(
                    signed_total_map,
                    origin="upper",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    transform=data_crs,
                )
            else:
                figure, axes = plt.subplots(figsize=(12, 5))
                image = axes.imshow(
                    signed_total_map,
                    origin="upper",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )

            if context_label:
                axes.set_title(f"Total input sensitivity (signed)\n{context_label}")
            else:
                axes.set_title("Total input sensitivity (signed)")
            axes.set_xlabel("Longitude")
            axes.set_ylabel("Latitude")
            self.maybe_add_coastlines(axes, color="black")
            self.add_target_area_patch(axes, data_crs=data_crs, edgecolor="black")
            self.apply_plot_limits(axes, data_crs=data_crs)
            figure.colorbar(image, ax=axes, shrink=0.8, label="Mean gradient")
            figure.tight_layout()
            figure.savefig(signed_total_path, dpi=150)
            plt.close(figure)
            paths.append(signed_total_path)

        top_count = min(self.owner.plot_top_k, len(top_channels))
        if top_count <= 0:
            return paths

        top_indices = [channel["index"] for channel in top_channels[:top_count]]
        columns = min(3, top_count)
        rows = math.ceil(top_count / columns)
        if projection is not None:
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(5 * columns, 3.5 * rows),
                squeeze=False,
                subplot_kw={"projection": projection},
            )
        else:
            figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)

        if context_label:
            figure.suptitle(f"Top input sensitivity channels\n{context_label}", fontsize=12)

        for axis in axes.ravel()[top_count:]:
            axis.axis("off")

        for axis, index in zip(axes.ravel(), top_indices):
            if projection is not None:
                image = axis.imshow(
                    np.abs(gradient_maps[index]),
                    origin="upper",
                    extent=extent,
                    cmap="viridis",
                    transform=data_crs,
                )
            else:
                image = axis.imshow(np.abs(gradient_maps[index]), origin="upper", extent=extent, cmap="viridis")
            axis.set_title(self.owner.ordering[index])
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            self.maybe_add_coastlines(axis, color="white")
            self.add_target_area_patch(axis, data_crs=data_crs, edgecolor="white")
            self.apply_plot_limits(axis, data_crs=data_crs)
            figure.colorbar(image, ax=axis, shrink=0.8)

        if context_label:
            figure.tight_layout(rect=(0, 0, 1, 0.9))
        else:
            figure.tight_layout()
        top_path = f"{self.owner.plot_prefix}-top-channels.png"
        figure.savefig(top_path, dpi=150)
        plt.close(figure)
        paths.append(top_path)

        if self.owner.plot_signed_gradients:
            signed_top_path = f"{self.owner.plot_prefix}-top-channels-signed.png"
            signed_vmin, signed_vmax = self.symmetric_limits(gradient_maps[top_indices])

            if projection is not None:
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(5 * columns, 3.5 * rows),
                    squeeze=False,
                    subplot_kw={"projection": projection},
                )
            else:
                figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)

            if context_label:
                figure.suptitle(f"Top input sensitivity channels (signed)\n{context_label}", fontsize=12)

            for axis in axes.ravel()[top_count:]:
                axis.axis("off")

            for axis, index in zip(axes.ravel(), top_indices):
                if projection is not None:
                    image = axis.imshow(
                        gradient_maps[index],
                        origin="upper",
                        extent=extent,
                        cmap="RdBu_r",
                        vmin=signed_vmin,
                        vmax=signed_vmax,
                        transform=data_crs,
                    )
                else:
                    image = axis.imshow(
                        gradient_maps[index],
                        origin="upper",
                        extent=extent,
                        cmap="RdBu_r",
                        vmin=signed_vmin,
                        vmax=signed_vmax,
                    )
                axis.set_title(f"{self.owner.ordering[index]} (signed)")
                axis.set_xlabel("Longitude")
                axis.set_ylabel("Latitude")
                self.maybe_add_coastlines(axis, color="black")
                self.add_target_area_patch(axis, data_crs=data_crs, edgecolor="black")
                self.apply_plot_limits(axis, data_crs=data_crs)
                figure.colorbar(image, ax=axis, shrink=0.8)

            if context_label:
                figure.tight_layout(rect=(0, 0, 1, 0.9))
            else:
                figure.tight_layout()
            figure.savefig(signed_top_path, dpi=150)
            plt.close(figure)
            paths.append(signed_top_path)

        return paths

    def write_sensitivity_netcdf(
        self,
        gradients,
        channel_scores,
        total_maps,
        objective_values,
        target_names,
        target_fields,
        target_metrics,
        target_areas,
    ):
        import xarray as xr

        target_count = len(target_names)
        target_area_array = np.full((target_count, 4), np.nan, dtype=np.float32)
        for index, area in enumerate(target_areas):
            if area is not None:
                target_area_array[index, :] = np.asarray(area, dtype=np.float32)

        dataset = xr.Dataset(
            data_vars={
                "sensitivity": (
                    ("target", "channel", "latitude", "longitude"),
                    gradients.astype(np.float32),
                ),
                "channel_score": (
                    ("target", "channel"),
                    channel_scores.astype(np.float32),
                ),
                "total_sensitivity": (
                    ("target", "latitude", "longitude"),
                    total_maps.astype(np.float32),
                ),
                "objective": (
                    ("target",),
                    np.asarray(objective_values, dtype=np.float32),
                ),
                "target_area": (
                    ("target", "area_coord"),
                    target_area_array,
                ),
            },
            coords={
                "target": np.asarray(target_names, dtype="U64"),
                "channel": np.asarray(self.owner.ordering, dtype="U16"),
                "latitude": np.asarray(self.owner.latitudes, dtype=np.float32),
                "longitude": np.asarray(self.owner.longitudes, dtype=np.float32),
                "area_coord": np.asarray(["north", "west", "south", "east"], dtype="U8"),
                "target_field": ("target", np.asarray(target_fields, dtype="U32")),
                "target_metric": ("target", np.asarray(target_metrics, dtype="U16")),
            },
            attrs={
                "model": self.model_name,
                "lead_time_hours": int(self.owner.lead_time),
                "generated_by": "ai-models sensitivity mode",
            },
        )

        dataset.to_netcdf(self.owner.sensitivity_path, engine="scipy")

    def save(self, gradients, objective_values):
        gradient_maps = np.stack([_channel_maps(gradient) for gradient in gradients], axis=0)
        channel_scores = np.stack([_channel_sensitivity_scores(gradient) for gradient in gradient_maps], axis=0)
        total_maps = np.stack([_total_sensitivity_map(gradient) for gradient in gradient_maps], axis=0)
        peaks = [self.total_sensitivity_peak(total_maps[index]) for index in range(len(self.targets))]

        target_names = [target.name for target in self.targets]
        target_fields = [target.field or "" for target in self.targets]
        target_metrics = [target.metric for target in self.targets]
        target_areas = [target.area for target in self.targets]

        self.write_sensitivity_netcdf(
            gradient_maps,
            channel_scores,
            total_maps,
            objective_values,
            target_names,
            target_fields,
            target_metrics,
            target_areas,
        )

        summary = {
            "lead_time_hours": self.owner.lead_time,
            "config_path": self.config_path,
            "targets": [],
        }

        for index, target in enumerate(self.targets):
            ranking = np.argsort(channel_scores[index])[::-1]
            top = [
                {
                    "channel": self.owner.ordering[channel_index],
                    "mean_abs_gradient": float(channel_scores[index, channel_index]),
                }
                for channel_index in ranking[:10]
            ]
            summary["targets"].append(
                {
                    "name": target.name,
                    "field": target.field,
                    "metric": target.metric,
                    "area": None
                    if target.area is None
                    else {
                        "north": target.area[0],
                        "west": target.area[1],
                        "south": target.area[2],
                        "east": target.area[3],
                    },
                    "objective": float(objective_values[index]),
                    "peak_total_sensitivity": peaks[index],
                    "top_channels": top,
                }
            )

        with open(self.owner.summary_path, "w") as file_handle:
            json.dump(summary, file_handle, indent=2)

        LOG.info("Saved sensitivities to %s", self.owner.sensitivity_path)
        LOG.info("Saved sensitivity summary to %s", self.owner.summary_path)

        for index, target in enumerate(self.targets):
            LOG.info(
                "Target %s peak total sensitivity at %.2fN %.2fE",
                target.name,
                peaks[index]["latitude"],
                peaks[index]["longitude"],
            )

        ranking = np.argsort(channel_scores[0])[::-1]
        top_channels = [
            {
                "index": int(channel_index),
                "channel": self.owner.ordering[channel_index],
                "mean_abs_gradient": float(channel_scores[0, channel_index]),
            }
            for channel_index in ranking[:10]
        ]
        LOG.info(
            "Top sensitivity channels (%s): %s",
            self.targets[0].name,
            ", ".join(channel["channel"] for channel in top_channels[:5]),
        )

        if self.owner.plot_sensitivity:
            base_prefix = self.owner.plot_prefix
            base_target = self.current_target
            for index, target in enumerate(self.targets):
                self.owner.plot_prefix = f"{base_prefix}-{target.name}"
                self.current_target = target
                ranking = np.argsort(channel_scores[index])[::-1]
                target_top = [
                    {
                        "index": int(channel_index),
                        "channel": self.owner.ordering[channel_index],
                        "mean_abs_gradient": float(channel_scores[index, channel_index]),
                    }
                    for channel_index in ranking[:10]
                ]
                plot_paths = self.write_sensitivity_plots(gradient_maps[index], target_top)
                LOG.info("Saved sensitivity plots for %s to %s", target.name, ", ".join(plot_paths))
            self.owner.plot_prefix = base_prefix
            self.current_target = base_target
