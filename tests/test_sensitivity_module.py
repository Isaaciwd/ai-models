import numpy as np

from ai_models.sensitivity import SensitivityManager
from ai_models.sensitivity import parse_target_area
from ai_models.sensitivity import signed_total_sensitivity_map
from ai_models.sensitivity import target_slug


def test_parse_target_area_with_string():
    assert parse_target_area("60,200,10,280") == (60.0, 200.0, 10.0, 280.0)


def test_parse_target_area_with_sequence():
    assert parse_target_area([50, 230, 30, 245]) == (50.0, 230.0, 30.0, 245.0)


def test_parse_target_area_wraps_longitudes():
    north, west, south, east = parse_target_area("50,-130,30,-115")
    assert np.isclose(north, 50.0)
    assert np.isclose(south, 30.0)
    assert np.isclose(west, 230.0)
    assert np.isclose(east, 245.0)


def test_target_slug_sanitizes():
    assert target_slug("West Coast R850", "fallback") == "west-coast-r850"


def test_signed_total_sensitivity_map_preserves_sign():
    gradient_maps = np.array(
        [
            [[1.0, -3.0], [2.0, -1.0]],
            [[-1.0, 1.0], [2.0, -3.0]],
        ],
        dtype=np.float32,
    )
    signed_map = signed_total_sensitivity_map(gradient_maps)
    expected = np.array([[0.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    assert np.allclose(signed_map, expected)


def test_load_config_applies_checkpointing_flags_from_run_section(tmp_path):
    class Owner:
        def __init__(self):
            self.model_checkpointing = None
            self.rollout_checkpointing = None
            self.plot_sensitivity = None
            self.plot_top_k = 6
            self.sensitivity_path = "sens.nc"
            self.summary_path = None
            self.plot_prefix = None
            self.plot_area = None
            self.sensitivity_metric = "mean-square"
            self.target_field = None
            self.target_param = None
            self.target_level = None
            self.target_area = None
            self.sensitivity = True
            self.sensitivity_config = None

        def default_target(self):
            return None

        def config_targets(self, config):
            return [None]

    owner = Owner()
    manager = SensitivityManager(owner=owner, model_name="x", default_sensitivity_path="sens.nc")
    manager.targets = [None]
    manager.current_target = None

    config_path = tmp_path / "sens.yaml"
    config_path.write_text(
        """
run:
  lead_time: 48
  model_checkpointing: false
  rollout_checkpointing: true
""".strip()
    )

    manager.load_config(str(config_path))

    assert owner.lead_time == 48
    assert owner.model_checkpointing is False
    assert owner.rollout_checkpointing is True


def test_load_config_accepts_null_plotting_area(tmp_path):
    class Owner:
        def __init__(self):
            self.model_checkpointing = None
            self.rollout_checkpointing = None
            self.plot_sensitivity = None
            self.plot_top_k = 6
            self.sensitivity_path = "sens.nc"
            self.summary_path = None
            self.plot_prefix = None
            self.plot_area = None
            self.sensitivity_metric = "mean-square"
            self.target_field = None
            self.target_param = None
            self.target_level = None
            self.target_area = None
            self.sensitivity = True
            self.sensitivity_config = None

        def default_target(self):
            return None

        def config_targets(self, config):
            return [None]

    owner = Owner()
    manager = SensitivityManager(owner=owner, model_name="x", default_sensitivity_path="sens.nc")
    manager.targets = [None]
    manager.current_target = None

    config_path = tmp_path / "sens.yaml"
    config_path.write_text(
        """
plotting:
  area: null
""".strip()
    )

    manager.load_config(str(config_path))

    assert manager.plot_area_bounds is None


def test_configure_enables_sensitivity_when_config_path_is_set(tmp_path):
    class Owner:
        def __init__(self):
            self.model_checkpointing = None
            self.rollout_checkpointing = None
            self.plot_sensitivity = None
            self.plot_signed_gradients = False
            self.plot_top_k = 6
            self.sensitivity_path = None
            self.summary_path = None
            self.plot_prefix = None
            self.plot_area = None
            self.sensitivity_metric = "mean-square"
            self.target_field = None
            self.target_param = None
            self.target_level = None
            self.target_area = None
            self.lead_time = 6
            self.sensitivity = False
            self.sensitivity_config = None

        def default_target(self):
            return None

        def config_targets(self, config):
            return [None]

    owner = Owner()
    config_path = tmp_path / "sens.yaml"
    config_path.write_text("{}")
    owner.sensitivity_config = str(config_path)
    manager = SensitivityManager(owner=owner, model_name="x", default_sensitivity_path="sens.nc")

    manager.configure()

    assert owner.sensitivity is True
    assert owner.model_checkpointing is True
    assert owner.rollout_checkpointing is True
