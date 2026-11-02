"""Pipeline tests: the physics generators, the loss definitions, the artifact.

    uv run pytest                      # all of it, a few seconds
    uv run pytest -k stellar           # one group
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import generate_data as gd


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# -- atmospheric generator -------------------------------------------------

def test_isa_temperature_spans_a_physical_range():
    """The lapse rate is applied to height, so T must span tens of K, not ~1.5."""
    t = gd.isa_temperature(np.array([800.0, 1013.25, 1050.0]))
    assert t[0] < t[1] < t[2]
    assert np.ptp(t) > 10.0
    # DATA.md 2.1 bounds, and sea-level pressure must give the ISA sea-level value.
    assert np.all((t > 230.0) & (t < 310.0))
    assert t[1] == pytest.approx(gd.T_SEA_LEVEL, abs=1e-6)


def test_synthetic_atmospheric_respects_declared_ranges(rng):
    x, y = gd._synthesize_atmospheric(rng, 4000)
    p, t, rh = x.T
    assert np.all((p >= 800.0) & (p <= 1050.0))
    assert np.all((rh >= 0.0) & (rh <= 100.0))
    assert np.all((t > 230.0) & (t < 310.0))
    assert np.isfinite(y).all()


def test_winds_satisfy_geostrophic_balance(rng):
    """u = -(1/rho f) dp/dy and v = (1/rho f) dp/dx, up to the additive noise."""
    x, y = gd._synthesize_atmospheric(rng, 2000)
    p, t, rh = x.T
    rho = gd.air_density(p, t)
    f = gd.coriolis_parameter()
    dpdx, dpdy = gd._synoptic_gradients(p, t, rh)

    assert y[:, 0] == pytest.approx(-dpdy / (rho * f), abs=8.0)
    assert y[:, 1] == pytest.approx(dpdx / (rho * f), abs=8.0)
    # Deeper lows drive stronger winds: the signal must dominate the noise.
    assert np.corrcoef(p, y[:, 0])[0, 1] < -0.9


def test_cleaning_drops_unphysical_rows_and_trims_tails(rng):
    x, y = gd._synthesize_atmospheric(rng, 1000)
    x[0] = [1013.0, 100.0, 50.0]   # T below 180 K
    x[1] = [1013.0, 288.0, -5.0]   # negative humidity
    y[2] = [np.nan, 0.0, 0.0]      # missing target
    x_clean, y_clean = gd._clean_regression(x, y)

    assert np.all(x_clean[:, 1] >= 180.0)
    assert np.all((x_clean[:, 2] >= 0.0) & (x_clean[:, 2] <= 100.0))
    assert np.isfinite(y_clean).all()
    # The 0.5% trim runs on six columns, so it removes a few percent of rows.
    assert 0.9 < x_clean.shape[0] / 997 < 1.0


# -- stellar generator -----------------------------------------------------

def test_stellar_classes_are_balanced_after_cuts(rng):
    x, labels = gd._synthesize_stellar(rng, 600)
    x, labels = gd._clean_stellar(x, labels, gd.N_PER_CLASS, rng)
    assert x.shape == (3 * gd.N_PER_CLASS, 3)
    assert [int((labels == c).sum()) for c in range(3)] == [gd.N_PER_CLASS] * 3
    for j, (lo, hi) in enumerate(gd.COLOR_LIMITS):
        assert np.all((x[:, j] >= lo) & (x[:, j] <= hi))


def test_white_dwarfs_are_the_blue_outliers(rng):
    """Separability is uneven by design: WDs stand apart, MS and giants overlap."""
    x, labels = gd._synthesize_stellar(rng, 2000)
    ms, rgb, wd = (x[labels == c] for c in range(3))
    assert wd[:, 0].mean() < ms[:, 0].mean() - 1.0   # much bluer in u-g
    assert wd[:, 1].mean() < rgb[:, 1].mean() - 0.5  # and in g-r
    # Giants sit redward of the main sequence in u-g, but not by much.
    offset = rgb[:, 0].mean() - gd._stellar_locus_ug(rgb[:, 1]).mean()
    assert 0.0 < offset < 0.5


def test_extinction_correction_dereddens(rng):
    colors = np.array([[1.5, 0.6, 0.3]])
    corrected = gd._extinction_correct(colors, np.array([0.1]))
    assert np.all(corrected < colors)        # de-reddening always makes colors bluer
    assert corrected[0, 0] == pytest.approx(1.5 - 0.0936, abs=1e-6)


def test_clean_stellar_raises_when_a_class_is_too_small(rng):
    x, labels = gd._synthesize_stellar(rng, 10)
    with pytest.raises(RuntimeError, match="Main Sequence"):
        gd._clean_stellar(x, labels, 500, rng)


# -- model and losses ------------------------------------------------------

def test_architecture_has_387_parameters():
    params = gd.export_parameters(gd.build_model(0))
    assert params["num_params"] == 387
    assert [(l["in_features"], l["out_features"]) for l in params["layers"]] == [
        (3, 16), (16, 16), (16, 3)
    ]
    assert sum(p.numel() for p in gd.build_model(0).parameters()) == 387


def test_regression_loss_is_mse_over_the_three_outputs():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    targets = torch.tensor([[1.0, 0.0, 6.0]])
    expected = (0.0 + 4.0 + 9.0) / 3.0
    assert gd.sample_losses(scores, targets, "regression").item() == pytest.approx(expected)


def test_classification_loss_is_cross_entropy():
    # Equal logits -> uniform softmax -> L = -log(1/3) for every class.
    scores = torch.zeros(3, 3)
    targets = torch.eye(3)
    losses = gd.sample_losses(scores, targets, "classification")
    assert losses.numpy() == pytest.approx(math.log(3.0), abs=1e-6)


def test_standardize_centers_and_scales():
    a = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    z, mean, std = gd.standardize(a)
    assert z[:, 0] == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert mean == pytest.approx([3.0, 10.0])
    assert std[1] == 1.0        # a constant column must not divide by zero
    assert np.isfinite(z).all()


def test_build_model_is_reproducible_and_seed_dependent():
    a = gd.export_parameters(gd.build_model(0))
    assert a == gd.export_parameters(gd.build_model(0))
    assert a != gd.export_parameters(gd.build_model(1))


def test_training_reduces_the_loss(rng):
    x, y = gd._synthesize_atmospheric(rng, 200)
    features = torch.tensor(gd.standardize(x)[0], dtype=torch.float32)
    targets = torch.tensor(gd.standardize(y)[0], dtype=torch.float32)

    model = gd.build_model(0)
    before = gd.sample_losses(model(features), targets, "regression").mean().item()
    gd.train_model(model, features, targets, "regression",
                   lr=0.05, batch_size=32, epochs=40, seed=7)
    after = gd.sample_losses(model(features), targets, "regression").mean().item()
    assert after < 0.5 * before


# -- the committed artifact ------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not gd.OUT_PATH.exists():
        pytest.skip(f"{gd.OUT_PATH} not built yet: run uv run python/generate_data.py")
    return json.loads(gd.OUT_PATH.read_text())


def test_artifact_matches_the_published_schema(artifact):
    datasets = artifact["datasets"]
    assert set(datasets) == {"atmospheric_regression", "stellar_classification"}

    for dataset in datasets.values():
        meta, data = dataset["metadata"], dataset["data"]
        n = meta["num_samples"]
        assert len(data["x"]) == len(data["y"]) == n
        assert all(len(row) == 3 for row in data["x"] + data["y"])
        assert len(meta["feature_names"]) == len(meta["target_names"]) == 3
        for key in ("features", "targets", "outputs"):
            assert all(lo <= hi for lo, hi in meta["quantiles"][key])

        ids = [w["id"] for w in dataset["weights"]]
        assert ids == ["random_1", "random_2", "random_3",
                       "trained_opt_1", "trained_opt_2", "trained_opt_3"]
        assert meta["default_weight_id"] in ids
        for entry in dataset["weights"]:
            assert entry["parameters"]["num_params"] == 387
            assert len(entry["outputs"]) == len(entry["sample_losses"]) == n
            assert entry["total_loss"] == pytest.approx(
                float(np.mean(entry["sample_losses"])), abs=1e-5
            )


def test_artifact_training_beats_random_initialization(artifact):
    for dataset in artifact["datasets"].values():
        by_id = {w["id"]: w for w in dataset["weights"]}
        worst_trained = max(by_id[f"trained_opt_{i}"]["total_loss"] for i in (1, 2, 3))
        best_random = min(by_id[f"random_{i}"]["total_loss"] for i in (1, 2, 3))
        assert worst_trained < 0.5 * best_random


def test_artifact_random_losses_sit_at_the_uninformed_baseline(artifact):
    """A random net knows nothing: MSE ~ 1 on standardized targets, CE ~ ln 3."""
    regression = artifact["datasets"]["atmospheric_regression"]["weights"]
    for entry in regression[:3]:
        assert entry["total_loss"] == pytest.approx(1.0, abs=0.35)

    classification = artifact["datasets"]["stellar_classification"]["weights"]
    for entry in classification[:3]:
        assert entry["total_loss"] == pytest.approx(math.log(3.0), abs=0.25)


def test_artifact_classification_outputs_are_probabilities(artifact):
    dataset = artifact["datasets"]["stellar_classification"]
    targets = np.array(dataset["data"]["y"])
    assert targets.sum(axis=1) == pytest.approx(np.ones(len(targets)))
    assert [int(c) for c in targets.sum(axis=0)] == [gd.N_PER_CLASS] * 3

    for entry in dataset["weights"]:
        probabilities = np.array(entry["outputs"])
        assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(probabilities)), abs=1e-3)
        assert np.all(probabilities >= 0.0)


def test_artifact_regression_outputs_are_in_physical_units(artifact):
    """Trained predictions must land on the targets' own scale, not the z-scored one."""
    dataset = artifact["datasets"]["atmospheric_regression"]
    targets = np.array(dataset["data"]["y"])
    outputs = np.array(
        next(w for w in dataset["weights"] if w["id"] == "trained_opt_3")["outputs"]
    )
    assert outputs.mean(axis=0) == pytest.approx(targets.mean(axis=0), abs=1.0)
    assert np.array(dataset["metadata"]["standardization"]["target_std"]) == pytest.approx(
        targets.std(axis=0), rel=1e-3
    )


# -- the real-data path ----------------------------------------------------
#
# The committed snapshot is synthetic, so without these the CSV loaders would
# be shipped unexecuted. Each one points RAW_DIR at a tmp directory, which is
# exactly how the pipeline discovers a real catalog.

def test_loaders_return_none_when_no_catalog_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "RAW_DIR", tmp_path)
    assert gd._load_atmospheric_raw() is None
    assert gd._load_stellar_raw() is None


def test_atmospheric_loader_reads_a_flattened_era5_export(tmp_path, monkeypatch):
    (tmp_path / "era5_sample.csv").write_text(
        "pressure_hpa,temperature_k,relative_humidity,u_ms,v_ms,w_cms\n"
        "1013.2,288.1,65.0,5.2,-3.1,0.4\n"
        "900.0,280.0,30.0,12.0,4.0,-0.2\n"
    )
    monkeypatch.setattr(gd, "RAW_DIR", tmp_path)
    x, y = gd._load_atmospheric_raw()

    # Column order is the contract: x = (p, T, RH), y = (u, v, w).
    assert x == pytest.approx(np.array([[1013.2, 288.1, 65.0], [900.0, 280.0, 30.0]]))
    assert y == pytest.approx(np.array([[5.2, -3.1, 0.4], [12.0, 4.0, -0.2]]))


def test_stellar_loader_reads_colors_and_classes(tmp_path, monkeypatch):
    (tmp_path / "stellar_colors.csv").write_text(
        "u_g,g_r,r_i,class\n"
        "1.12,0.45,0.18,0\n"
        "2.40,0.90,0.35,1\n"
        "0.20,-0.15,-0.10,2\n"
    )
    monkeypatch.setattr(gd, "RAW_DIR", tmp_path)
    x, labels = gd._load_stellar_raw()

    assert x == pytest.approx(
        np.array([[1.12, 0.45, 0.18], [2.40, 0.90, 0.35], [0.20, -0.15, -0.10]])
    )
    assert labels.tolist() == [0, 1, 2]
    assert labels.dtype.kind == "i"


def test_stellar_loader_applies_extinction_when_ebv_is_supplied(tmp_path, monkeypatch):
    (tmp_path / "stellar_colors.csv").write_text(
        "u_g,g_r,r_i,class,ebv\n"
        "1.12,0.45,0.18,0,0.10\n"
        "2.40,0.90,0.35,1,0.00\n"
    )
    monkeypatch.setattr(gd, "RAW_DIR", tmp_path)
    x, _ = gd._load_stellar_raw()

    # The reddened row is de-reddened; the E(B-V) = 0 row passes through.
    assert x[0] == pytest.approx([1.12 - 0.0936, 0.45 - 0.1018, 0.18 - 0.0587], abs=1e-6)
    assert x[1] == pytest.approx([2.40, 0.90, 0.35])


def _write_regression_catalog(path, rng, n=700):
    x, y = gd._synthesize_atmospheric(rng, n)
    rows = ["pressure_hpa,temperature_k,relative_humidity,u_ms,v_ms,w_cms"]
    rows += [",".join(f"{v:.6f}" for v in (*xi, *yi)) for xi, yi in zip(x, y)]
    path.write_text("\n".join(rows) + "\n")


def _write_stellar_catalog(path, rng, n_per_class=700):
    x, labels = gd._synthesize_stellar(rng, n_per_class)
    rows = ["u_g,g_r,r_i,class"]
    rows += [",".join([*(f"{v:.6f}" for v in xi), str(int(c))]) for xi, c in zip(x, labels)]
    path.write_text("\n".join(rows) + "\n")


def test_regression_dataset_uses_a_catalog_when_one_exists(tmp_path, monkeypatch, rng):
    """End to end on the real-data branch: the catalog must win over the generator."""
    _write_regression_catalog(tmp_path / "era5_sample.csv", rng)
    monkeypatch.setattr(gd, "RAW_DIR", tmp_path)

    dataset = gd.atmospheric_dataset(epochs=1)
    meta = dataset["metadata"]
    assert meta["source"] == "era5"          # not "synthetic"
    assert meta["num_samples"] == gd.N_REGRESSION
    assert len(dataset["weights"]) == 6
    assert np.all(np.array(dataset["data"]["x"])[:, 0] >= 800.0)


def test_classification_dataset_uses_a_catalog_when_one_exists(tmp_path, monkeypatch, rng):
    _write_stellar_catalog(tmp_path / "stellar_colors.csv", rng)
    monkeypatch.setattr(gd, "RAW_DIR", tmp_path)

    dataset = gd.stellar_dataset(epochs=1)
    meta = dataset["metadata"]
    assert meta["source"] == "sdss"
    assert meta["num_samples"] == 3 * gd.N_PER_CLASS
    targets = np.array(dataset["data"]["y"])
    assert [int(c) for c in targets.sum(axis=0)] == [gd.N_PER_CLASS] * 3


def test_committed_snapshot_is_synthetic(artifact):
    """The snapshot in the repo is generated, not observed. If a real catalog is
    ever used, this is the test that should change."""
    sources = {k: v["metadata"]["source"] for k, v in artifact["datasets"].items()}
    assert sources == {
        "atmospheric_regression": "synthetic",
        "stellar_classification": "synthetic",
    }
