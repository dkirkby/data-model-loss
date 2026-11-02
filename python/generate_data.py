"""Build the single static artifact the loss-function visualizer reads.

Run from anywhere in the repo (paths resolve from this file, not the cwd):

    uv run python/generate_data.py            # regenerate the snapshot
    uv run python/generate_data.py --epochs 1200

Output: ``frontend/public/data/model_runs.json`` -- see DATA.md section 4 for the
schema. The pipeline is offline by design: it looks for real catalog files in
``python/raw_data/`` and falls back to physics-based synthetic generators when
they are absent, so a clean checkout can build the site with no network access.

Units convention: ``data.x``, ``data.y`` and (for regression) ``weights[].outputs``
are in PHYSICAL units so the corner plot can label real axes. The network trains on
standardized values, so ``sample_losses`` and ``total_loss`` live in that
standardized space; ``metadata.standardization`` carries the constants needed to
move between the two.

Deviations from DATA.md, all flagged where they occur below:
  * The regression temperature applies the ISA lapse rate to geopotential HEIGHT
    (via the hypsometric relation) rather than to a pressure difference in hPa, so
    that T spans a physical range instead of a ~1.5 K sliver. See `isa_temperature`.
  * Geostrophic balance needs a pressure GRADIENT, which a per-sample scalar p
    cannot supply. `_synoptic_gradients` builds one from the state variables.
  * The vertical-velocity formula is used exactly as written, including its sign.
    See the note in `_synthesize_atmospheric`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_DIR = HERE / "raw_data"
OUT_PATH = ROOT / "frontend" / "public" / "data" / "model_runs.json"

# --------------------------------------------------------------------------
# Physical constants (atmospheric task)
# --------------------------------------------------------------------------

OMEGA = 7.2921e-5        # Earth's rotation rate, rad/s
R_DRY = 287.05           # gas constant for dry air, J/(kg K)
LATITUDE_DEG = 45.0      # mid-latitude reference, where f is evaluated
P_SEA_LEVEL = 1013.25    # ISA sea-level pressure, hPa
T_SEA_LEVEL = 288.15     # ISA sea-level temperature, K
LAPSE_RATE = 0.0065      # ISA tropospheric lapse rate, K/m

# Sampling / cleaning
N_REGRESSION = 500       # samples kept for the regression task
N_PER_CLASS = 500        # samples kept per stellar class (1500 total)
TRIM_Q = 0.005           # DATA.md 2.3: trim the top and bottom 0.5% tails
PLOT_Q = (0.005, 0.995)  # PLAN.md 4A: corner-plot axis limits

# Model and training (PLAN.md 3, DATA.md 5)
ARCH = (3, 16, 16, 3)
RANDOM_SEEDS = (0, 1, 2)
TRAIN_CONFIGS = (
    {"lr": 0.001, "batch_size": 32, "seed": 10},
    {"lr": 0.010, "batch_size": 32, "seed": 11},
    {"lr": 0.100, "batch_size": 64, "seed": 12},
)
DEFAULT_EPOCHS = 800

# Serialized precision: enough to plot and to reproduce the losses, small enough
# that the artifact stays a comfortable download.
DP_DATA = 4
DP_LOSS = 6
DP_PARAM = 6


# --------------------------------------------------------------------------
# International Standard Atmosphere helpers
# --------------------------------------------------------------------------

def isa_height(p_hpa: np.ndarray) -> np.ndarray:
    """Geopotential height of a pressure level, m (inverse hypsometric relation)."""
    return 44330.77 * (1.0 - (p_hpa / P_SEA_LEVEL) ** 0.190263)


def isa_temperature(p_hpa: np.ndarray) -> np.ndarray:
    """Standard-atmosphere temperature at a pressure level, K.

    DATA.md writes ``T = 288.15 - 0.0065*(1013.25 - p)``, applying the lapse rate
    directly to a pressure difference in hPa. 0.0065 is K/m, so that expression
    spans only ~1.5 K across the whole pressure range and leaves temperature
    almost pure noise. Converting pressure to height first is what the formula
    intends and gives T ~ 275-290 K, inside DATA.md's stated [230, 310] bound.
    """
    return T_SEA_LEVEL - LAPSE_RATE * isa_height(p_hpa)


def coriolis_parameter(latitude_deg: float = LATITUDE_DEG) -> float:
    """Coriolis parameter f = 2 Omega sin(phi), 1/s."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(latitude_deg))


def air_density(p_hpa: np.ndarray, t_kelvin: np.ndarray) -> np.ndarray:
    """Ideal-gas density, kg/m^3 (p converted from hPa to Pa)."""
    return (p_hpa * 100.0) / (R_DRY * t_kelvin)


# --------------------------------------------------------------------------
# Regression task: 3D wind vectors from thermodynamic state
# --------------------------------------------------------------------------

def _synoptic_gradients(
    p_hpa: np.ndarray, t_kelvin: np.ndarray, rh_pct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal pressure-gradient components (Pa/m) implied by the local state.

    Geostrophic balance relates wind to grad(p), but a sample carries only a
    scalar pressure, so the gradient has to be reconstructed. Two synoptic
    regularities do that while keeping the targets a genuine (noisy) function of
    the features, which is what makes the task learnable:

      * Meridional gradient: the climatological mid-latitude westerly background
        plus cyclonic tightening. Gradients steepen with the depth of a pressure
        anomaly, and more steeply for lows than for highs -- real cyclones are
        tighter than anticyclones -- which makes the map nonlinear and asymmetric.
      * Zonal gradient: warm-sector flow. A warm anomaly relative to the standard
        atmosphere at the same pressure sits ahead of a trough, where the flow is
        southerly (v > 0); moist air strengthens it.
    """
    rho = air_density(p_hpa, t_kelvin)
    f = coriolis_parameter()

    depth = (P_SEA_LEVEL - p_hpa) / 50.0            # >0 for lows, <0 for highs
    thermal = t_kelvin - isa_temperature(p_hpa)      # warm anomaly, K
    moisture = rh_pct / 100.0 - 0.5                  # moisture anomaly, +-0.5

    u_geo = 5.0 + 3.0 * depth * (1.0 + 0.35 * np.clip(depth, 0.0, None))
    v_geo = 4.0 * thermal * (1.0 + 0.6 * moisture)

    dpdy = -rho * f * u_geo
    dpdx = rho * f * v_geo
    return dpdx, dpdy


def _synthesize_atmospheric(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Physics-based synthetic generator for the wind-vector task (DATA.md 2.2)."""
    p = rng.uniform(800.0, 1050.0, size=n)
    t = isa_temperature(p) + rng.normal(0.0, 2.0, size=n)
    rh = 100.0 * rng.beta(2.0, 2.0, size=n)

    rho = air_density(p, t)
    f = coriolis_parameter()
    dpdx, dpdy = _synoptic_gradients(p, t, rh)

    u = -dpdy / (rho * f) + rng.normal(0.0, 1.5, size=n)
    v = dpdx / (rho * f) + rng.normal(0.0, 1.5, size=n)
    # DATA.md's w, used verbatim. Note the sign: warm, moist air gives w < 0 here,
    # which reads as subsidence for a geometric vertical velocity but matches the
    # reanalysis omega = dp/dt convention, where negative means ascent. One
    # consequence to be aware of: T > 273.15 K across the sampled pressure
    # range, so w comes out almost entirely one-signed. The magnitude, a
    # fraction of a cm/s, is the right order for synoptic-scale vertical motion.
    w = -(rh / 100.0) * ((t - 273.15) / 50.0) + rng.normal(0.0, 0.03, size=n)

    x = np.column_stack([p, t, rh])
    y = np.column_stack([u, v, w])
    return x, y


def _load_atmospheric_raw() -> tuple[np.ndarray, np.ndarray] | None:
    """Load an ERA5 sample from ``python/raw_data/`` if one is present.

    Expects ``era5_sample.csv`` with a header row naming ``pressure_hpa``,
    ``temperature_k``, ``relative_humidity``, ``u_ms``, ``v_ms``, ``w_cms``.
    A NetCDF download from the Copernicus CDS (``cdsapi``) must be flattened to
    that form first, which keeps xarray/netCDF4 out of the dependency set.
    """
    path = RAW_DIR / "era5_sample.csv"
    if not path.exists():
        return None
    table = np.genfromtxt(path, delimiter=",", names=True)
    x = np.column_stack(
        [table["pressure_hpa"], table["temperature_k"], table["relative_humidity"]]
    )
    y = np.column_stack([table["u_ms"], table["v_ms"], table["w_cms"]])
    return x, y


def _clean_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask unphysical rows, then trim the 0.5% tails (DATA.md 2.3)."""
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    physical = (x[:, 2] >= 0.0) & (x[:, 2] <= 100.0) & (x[:, 1] >= 180.0)
    x, y = x[finite & physical], y[finite & physical]

    columns = np.column_stack([x, y])
    lo = np.quantile(columns, TRIM_Q, axis=0)
    hi = np.quantile(columns, 1.0 - TRIM_Q, axis=0)
    keep = ((columns >= lo) & (columns <= hi)).all(axis=1)
    return x[keep], y[keep]


# --------------------------------------------------------------------------
# Classification task: stellar spectral class from optical colors
# --------------------------------------------------------------------------

CLASS_NAMES = ("Main Sequence", "Red Giant", "White Dwarf")

# SDSS extinction coefficients A_band / E(B-V) (Schlegel, Finkbeiner & Davis).
# Differenced into the three color indices used as features.
EXTINCTION_UG = 4.239 - 3.303
EXTINCTION_GR = 3.303 - 2.285
EXTINCTION_RI = 2.285 - 1.698

# Sanity cuts in color space (DATA.md 3.3, "color cuts").
COLOR_LIMITS = ((-0.6, 4.0), (-0.5, 2.0), (-0.5, 2.5))


def _stellar_locus_ug(gr: np.ndarray) -> np.ndarray:
    """u-g along the SDSS main-sequence stellar locus, as a function of g-r."""
    return 0.78 + 1.95 * gr - 0.38 * gr**2


def _stellar_locus_ri(gr: np.ndarray) -> np.ndarray:
    """r-i along the main-sequence locus: shallow for warm stars, steep for M dwarfs."""
    return 0.33 * gr + 0.42 * gr**3


def _synthesize_stellar(
    rng: np.random.Generator, n_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Physics-based synthetic generator for the color-classification task.

    The three classes are curved loci rather than isotropic blobs, so the
    boundaries between them are nonlinear (DATA.md 3.2). Their separability is
    deliberately uneven, and that is the astrophysics: white dwarfs are hot and
    sit well away from everything else in color space, while giants and dwarfs
    overlap, because broadband colors measure temperature and barely constrain
    surface gravity. A trained model should therefore reach near-zero loss on
    white dwarfs and keep a visible residual confusion between the other two.
    """
    # Main sequence: the full locus, from blue A stars to red M dwarfs.
    t = rng.uniform(0.0, 1.0, size=n_per_class)
    gr_ms = -0.25 + 1.65 * t
    ms = np.column_stack([_stellar_locus_ug(gr_ms), gr_ms, _stellar_locus_ri(gr_ms)])
    ms += rng.normal(0.0, 0.06, size=ms.shape)

    # Red giant branch: cool half of the locus, offset redward in u-g (stronger
    # metal-line blanketing at low surface gravity) and bluer in r-i.
    t = rng.beta(2.0, 2.0, size=n_per_class)
    gr_rgb = 0.45 + 0.75 * t
    rgb = np.column_stack(
        [_stellar_locus_ug(gr_rgb) + 0.18, gr_rgb, 0.30 * gr_rgb + 0.10]
    )
    rgb += rng.normal(0.0, 0.12, size=rgb.shape)

    # White dwarfs: hot, compact, blue.
    wd = np.column_stack([
        rng.normal(0.28, 0.16, size=n_per_class),
        rng.normal(-0.16, 0.13, size=n_per_class),
        rng.normal(-0.11, 0.10, size=n_per_class),
    ])

    x = np.vstack([ms, rgb, wd])
    labels = np.repeat(np.arange(3), n_per_class)
    return x, labels


def _load_stellar_raw() -> tuple[np.ndarray, np.ndarray] | None:
    """Load an SDSS/Gaia color catalog from ``python/raw_data/`` if present.

    Expects ``stellar_colors.csv`` with ``u_g``, ``g_r``, ``r_i``, ``class``
    (0 = main sequence, 1 = red giant, 2 = white dwarf) and optionally ``ebv``,
    the reddening used for the extinction correction below.
    """
    path = RAW_DIR / "stellar_colors.csv"
    if not path.exists():
        return None
    table = np.genfromtxt(path, delimiter=",", names=True)
    x = np.column_stack([table["u_g"], table["g_r"], table["r_i"]])
    if "ebv" in (table.dtype.names or ()):
        x = _extinction_correct(x, table["ebv"])
    return x, table["class"].astype(int)


def _extinction_correct(colors: np.ndarray, ebv: np.ndarray) -> np.ndarray:
    """De-redden observed color indices for Galactic dust (DATA.md 3.3)."""
    correction = np.column_stack([
        EXTINCTION_UG * ebv, EXTINCTION_GR * ebv, EXTINCTION_RI * ebv
    ])
    return colors - correction


def _clean_stellar(
    x: np.ndarray, labels: np.ndarray, n_per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply color cuts, then balance to exactly ``n_per_class`` per class."""
    keep = np.isfinite(x).all(axis=1)
    for j, (lo, hi) in enumerate(COLOR_LIMITS):
        keep &= (x[:, j] >= lo) & (x[:, j] <= hi)
    x, labels = x[keep], labels[keep]

    chosen = []
    for cls in range(3):
        idx = np.flatnonzero(labels == cls)
        if idx.size < n_per_class:
            raise RuntimeError(
                f"class {cls} ({CLASS_NAMES[cls]}) has only {idx.size} samples "
                f"after color cuts, need {n_per_class}"
            )
        chosen.append(rng.choice(idx, size=n_per_class, replace=False))
    order = np.sort(np.concatenate(chosen))
    return x[order], labels[order]


# --------------------------------------------------------------------------
# Model, training, evaluation
# --------------------------------------------------------------------------

def _rounded(a: np.ndarray, decimals: int) -> list:
    """Round for serialization, widening from float32 first.

    Rounding a float32 and letting JSON widen it to float64 leaves artifacts
    (1.36111 -> 1.361109972000122), which both bloats the artifact and makes it
    unreadable. The cast is what actually produces short decimals.
    """
    return np.round(np.asarray(a, dtype=np.float64), decimals).tolist()


def standardize(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column, returning the standardized array plus mean and sd."""
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return (a - mean) / std, mean, std


def build_model(seed: int) -> nn.Sequential:
    """The fixed 3 -> 16 -> 16 -> 3 dense network (PLAN.md 3), ReLU hidden layers.

    The final layer emits raw scores: identity is the regression output
    activation, and the classifier's softmax is folded into the cross-entropy
    loss and applied explicitly only when exporting probabilities.
    """
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for i in range(len(ARCH) - 1):
        layers.append(nn.Linear(ARCH[i], ARCH[i + 1]))
        if i < len(ARCH) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def sample_losses(scores: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
    """Per-sample loss L(yhat_i, y_i; Theta) in the model's own output space."""
    if task == "regression":
        # MSE averaged over the three outputs (PLAN.md 3).
        return ((scores - targets) ** 2).mean(dim=1)
    # Categorical cross-entropy against one-hot targets, from logits.
    return F.cross_entropy(scores, targets.argmax(dim=1), reduction="none")


def train_model(
    model: nn.Sequential,
    features: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> nn.Sequential:
    """Plain SGD, as specified -- no momentum, no schedule, no validation split."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = sample_losses(model(features[batch]), targets[batch], task).mean()
            loss.backward()
            optimizer.step()
    return model


def export_parameters(model: nn.Sequential) -> dict:
    """Serialize Theta layer by layer, with the total parameter count."""
    layers = []
    num_params = 0
    for module in model:
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach().numpy()
        bias = module.bias.detach().numpy()
        num_params += weight.size + bias.size
        layers.append({
            "in_features": module.in_features,
            "out_features": module.out_features,
            "weights": _rounded(weight, DP_PARAM),
            "biases": _rounded(bias, DP_PARAM),
        })
    return {"num_params": int(num_params), "layers": layers}


def evaluate(
    model: nn.Sequential,
    features: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass -> (outputs in display units, per-sample losses in train units)."""
    with torch.no_grad():
        scores = model(features)
        losses = sample_losses(scores, targets, task).numpy()
        if task == "regression":
            # Back to physical units (m/s, cm/s) so the corner plot can label axes.
            outputs = scores.numpy() * target_std + target_mean
        else:
            # Class probabilities, directly comparable to the one-hot targets.
            outputs = F.softmax(scores, dim=1).numpy()
    return outputs, losses


def build_weight_sets(
    features: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    epochs: int,
) -> list[dict]:
    """The six weight sets: three random initializations, three SGD runs."""
    entries = []

    for i, seed in enumerate(RANDOM_SEEDS, start=1):
        model = build_model(seed)
        outputs, losses = evaluate(model, features, targets, task, target_mean, target_std)
        entries.append({
            "id": f"random_{i}",
            "name": f"Random Init #{i}",
            "is_trained": False,
            "hyperparameters": {"seed": seed},
            "parameters": export_parameters(model),
            "outputs": _rounded(outputs, DP_DATA),
            "sample_losses": _rounded(losses, DP_LOSS),
            "total_loss": round(float(np.mean(losses, dtype=np.float64)), DP_LOSS),
        })

    for i, config in enumerate(TRAIN_CONFIGS, start=1):
        model = build_model(config["seed"])
        train_model(
            model, features, targets, task,
            lr=config["lr"], batch_size=config["batch_size"],
            epochs=epochs, seed=config["seed"],
        )
        outputs, losses = evaluate(model, features, targets, task, target_mean, target_std)
        entries.append({
            "id": f"trained_opt_{i}",
            "name": f"SGD lr={config['lr']:g}, batch {config['batch_size']} ({epochs} epochs)",
            "is_trained": True,
            "hyperparameters": {**config, "epochs": epochs, "optimizer": "SGD"},
            "parameters": export_parameters(model),
            "outputs": _rounded(outputs, DP_DATA),
            "sample_losses": _rounded(losses, DP_LOSS),
            "total_loss": round(float(np.mean(losses, dtype=np.float64)), DP_LOSS),
        })

    return entries


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def _quantile_ranges(a: np.ndarray) -> list[list[float]]:
    """Per-column [0.5%, 99.5%] range, the corner plot's axis limits."""
    lo = np.quantile(a, PLOT_Q[0], axis=0)
    hi = np.quantile(a, PLOT_Q[1], axis=0)
    return _rounded(np.column_stack([lo, hi]), DP_DATA)


def build_dataset(
    *,
    domain: str,
    task_type: str,
    feature_names: list[str],
    target_names: list[str],
    loss_name: str,
    source: str,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
) -> dict:
    """Preprocess, train the six weight sets, and assemble one dataset entry."""
    x_std, x_mean, x_sd = standardize(x)
    if task_type == "regression":
        # DATA.md 2.3: standardize the targets too, so MSE weighs u, v and w
        # equally instead of being dominated by the two in m/s.
        y_std, y_mean, y_sd = standardize(y)
    else:
        y_std = y
        y_mean = np.zeros(y.shape[1])
        y_sd = np.ones(y.shape[1])

    features = torch.tensor(x_std, dtype=torch.float32)
    targets = torch.tensor(y_std, dtype=torch.float32)
    weights = build_weight_sets(features, targets, task_type, y_mean, y_sd, epochs)

    all_outputs = np.concatenate([np.asarray(w["outputs"]) for w in weights], axis=0)

    return {
        "metadata": {
            "domain": domain,
            "task_type": task_type,
            "loss_name": loss_name,
            "source": source,
            "feature_names": feature_names,
            "target_names": target_names,
            "num_samples": int(x.shape[0]),
            "default_weight_id": "trained_opt_1",
            "quantiles": {
                "features": _quantile_ranges(x),
                "targets": _quantile_ranges(y),
                # Across all six weight sets, so the y/yhat histograms on the
                # bottom-right diagonal can share one axis without clipping a
                # random initialization off-screen.
                "outputs": _quantile_ranges(all_outputs),
            },
            "standardization": {
                "feature_mean": _rounded(x_mean, DP_PARAM),
                "feature_std": _rounded(x_sd, DP_PARAM),
                "target_mean": _rounded(y_mean, DP_PARAM),
                "target_std": _rounded(y_sd, DP_PARAM),
            },
        },
        "data": {
            "x": _rounded(x, DP_DATA),
            "y": _rounded(y, DP_DATA),
        },
        "weights": weights,
    }


def atmospheric_dataset(epochs: int, seed: int = 20260101) -> dict:
    """Regression task: 3D wind vectors from pressure, temperature and humidity."""
    rng = np.random.default_rng(seed)
    raw = _load_atmospheric_raw()
    if raw is None:
        # Oversample so the 0.5% tail trim still leaves N_REGRESSION rows.
        x, y = _synthesize_atmospheric(rng, int(N_REGRESSION * 1.3))
        source = "synthetic"
    else:
        x, y = raw
        source = "era5"
    x, y = _clean_regression(x, y)
    if x.shape[0] < N_REGRESSION:
        raise RuntimeError(f"only {x.shape[0]} rows survived cleaning, need {N_REGRESSION}")
    x, y = x[:N_REGRESSION], y[:N_REGRESSION]

    return build_dataset(
        domain="Atmospheric Science",
        task_type="regression",
        feature_names=["Pressure (hPa)", "Temperature (K)", "Humidity (%)"],
        target_names=["u-wind (m/s)", "v-wind (m/s)", "w-wind (cm/s)"],
        loss_name="mse",
        source=source,
        x=x, y=y, epochs=epochs,
    )


def stellar_dataset(epochs: int, seed: int = 20260102) -> dict:
    """Classification task: stellar spectral class from optical color indices."""
    rng = np.random.default_rng(seed)
    raw = _load_stellar_raw()
    if raw is None:
        x, labels = _synthesize_stellar(rng, int(N_PER_CLASS * 1.2))
        source = "synthetic"
    else:
        x, labels = raw
        source = "sdss"
    x, labels = _clean_stellar(x, labels, N_PER_CLASS, rng)
    y = np.eye(3)[labels]

    return build_dataset(
        domain="Astrophysics",
        task_type="classification",
        feature_names=["u - g", "g - r", "r - i"],
        target_names=list(CLASS_NAMES),
        loss_name="cross_entropy",
        source=source,
        x=x, y=y, epochs=epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"SGD epochs for each trained weight set (default {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_PATH, help="output JSON path",
    )
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    datasets = {
        "atmospheric_regression": atmospheric_dataset(args.epochs),
        "stellar_classification": stellar_dataset(args.epochs),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"datasets": datasets}, separators=(",", ":")) + "\n")

    size_kb = args.out.stat().st_size / 1024.0
    print(f"wrote {args.out.relative_to(ROOT)} ({size_kb:.0f} KB)")
    for key, dataset in datasets.items():
        meta = dataset["metadata"]
        print(f"\n{key}  [{meta['source']}, N={meta['num_samples']}, {meta['loss_name']}]")
        for entry in dataset["weights"]:
            print(f"  {entry['id']:<15} total_loss={entry['total_loss']:<12.6f} {entry['name']}")


if __name__ == "__main__":
    main()
