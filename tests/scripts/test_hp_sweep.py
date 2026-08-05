"""Tests for the HP sweep driver (scripts/hp_sweep.py)."""

import os
import sys
import tempfile

import pytest
import yaml

# Make the scripts/ dir importable in tests.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

import hp_sweep


@pytest.fixture
def grid_yaml(tmp_path):
    path = tmp_path / "grid.yaml"
    spec = {
        "model": "tdt",
        "task": "B",
        "method": "grid",
        "params": {
            "lr": [1e-3, 1e-4],
            "d_model": [64, 128],
        },
    }
    path.write_text(yaml.safe_dump(spec))
    return str(path)


@pytest.fixture
def random_yaml(tmp_path):
    path = tmp_path / "random.yaml"
    spec = {
        "model": "tdt",
        "task": "B",
        "method": "random",
        "n_trials": 5,
        "seed": 7,
        "params": {
            "lr": {"log_min": 1e-4, "log_max": 1e-2},
            "d_model": [64, 128, 256],
            "dropout": {"min": 0.0, "max": 0.3},
        },
    }
    path.write_text(yaml.safe_dump(spec))
    return str(path)


class TestGrid:

    def test_enumerates_cartesian_product(self, grid_yaml):
        seen = set()
        for trial in range(4):  # 2 * 2 = 4
            cfg = hp_sweep.sample_config(trial, grid_yaml)
            seen.add((cfg["lr"], cfg["d_model"]))
        assert len(seen) == 4

    def test_out_of_range_trial_raises(self, grid_yaml):
        with pytest.raises(IndexError):
            hp_sweep.sample_config(99, grid_yaml)


class TestRandom:

    def test_deterministic_for_same_seed(self, random_yaml):
        a = hp_sweep.sample_config(2, random_yaml)
        b = hp_sweep.sample_config(2, random_yaml)
        assert a == b

    def test_different_trials_differ(self, random_yaml):
        a = hp_sweep.sample_config(0, random_yaml)
        b = hp_sweep.sample_config(1, random_yaml)
        assert a != b

    def test_log_uniform_in_range(self, random_yaml):
        for trial in range(5):
            cfg = hp_sweep.sample_config(trial, random_yaml)
            assert 1e-4 <= cfg["lr"] <= 1e-2

    def test_uniform_in_range(self, random_yaml):
        for trial in range(5):
            cfg = hp_sweep.sample_config(trial, random_yaml)
            assert 0.0 <= cfg["dropout"] <= 0.3

    def test_categorical_value_from_list(self, random_yaml):
        choices = {64, 128, 256}
        for trial in range(5):
            cfg = hp_sweep.sample_config(trial, random_yaml)
            assert cfg["d_model"] in choices


class TestCLITranslation:

    def test_known_keys_use_canonical_flags(self):
        cli = hp_sweep._cfg_to_cli_args(
            {"lr": 1e-3, "batch_size": 64, "d_model": 128, "e_layers": 4},
            model="tdt", task="B", extra=[],
        )
        # d_model maps to --state-size; e_layers maps to --lstm-layers.
        assert "--lr" in cli and "1e-3" in [x.lower() for x in cli] or "0.001" in cli
        assert "--batch-size" in cli
        assert "--state-size" in cli
        assert "--lstm-layers" in cli

    def test_unknown_keys_passed_through(self):
        cli = hp_sweep._cfg_to_cli_args(
            {"some_new_arg": "value"},
            model="tdt", task="B", extra=[],
        )
        assert "--some-new-arg" in cli
