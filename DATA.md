# DATA.md: Scientific Data Acquisition and Preprocessing Specification

This document details the acquisition, synthetic physical modeling fallbacks, preprocessing steps, and serialization formats for datasets used in the Loss Functions and Optimization visualizer.

## 1. Datasets Overview

Each dataset consists of 3 input features ($\mathbf{x}_i \in \mathbb{R}^3$) and 3 target outputs ($\mathbf{y}_i \in \mathbb{R}^3$). To ensure zero external API dependencies while maintaining scientific fidelity, each dataset supports both **Real-World Catalog/Reanalysis Data** and **Physics-Based Synthetic Generators**.

### Provenance of the committed snapshot

**Both datasets in `frontend/public/data/model_runs.json` are synthetic.** No real
catalog has ever been placed in `python/raw_data/`, which does not exist in the
repository: every number in the snapshot comes from the physics-based generators in
§2.2 and §3.2. `metadata.source` records this as `"synthetic"` for each dataset, the
generator script prints it on every run, and Panel C displays it (PLAN.md §4C) so the
distinction reaches the lecture room.

The real-data loaders are implemented and tested against fixture catalogs, but they have
never been run on an actual ERA5 or SDSS export. Supplying one changes more than the
numbers: real data has genuine gaps for the masks in §2.3 to catch, and its targets stop
being a deterministic function of its features, so the trained losses in §5 would floor
well above the values tabulated there.

## 2. Regression Task: 3D Wind Vector Prediction (Atmospheric Science)

### 2.1. Physical Problem Formulation

Inferring 3D atmospheric velocity fields $(u, v, w)$ from local thermodynamic state variables under approximate geostrophic and hydrostatic balances.

* **Feature Vector (**$\mathbf{x} \in \mathbb{R}^3$**)**:

  1. $x_1$: Atmospheric Pressure $p$ ($\text{hPa}$) — range: $[800, 1050]$

  2. $x_2$: Air Temperature $T$ ($\text{K}$) — range: $[230, 310]$

  3. $x_3$: Relative Humidity $RH$ ($\%$) — range: $[0, 100]$

* **Target Vector (**$\mathbf{y} \in \mathbb{R}^3$**)**:

  1. $y_1$: Zonal (East-West) Wind $u$ ($\text{m/s}$)

  2. $y_2$: Meridional (North-South) Wind $v$ ($\text{m/s}$)

  3. $y_3$: Vertical Velocity $w$ ($\text{cm/s}$)

### 2.2. Data Sources & Acquisition

* **Real-World Dataset**: ERA5 Reanalysis sample via ECMWF Copernicus Climate Data Store (`cdsapi`).

* **Synthetic Fallback Generator**:
  

  $$
  \mathbf{x}_1 \sim \text{Uniform}(800, 1050)
  $$

  $$
  z = 44330.77 \left(1 - \left(\tfrac{\mathbf{x}_1}{1013.25}\right)^{0.190263}\right), \quad
  \mathbf{x}_2 = 288.15 - 0.0065 \cdot z + \epsilon_T, \quad \epsilon_T \sim \mathcal{N}(0, 2^2)
  $$

  The ISA lapse rate is $0.0065\,\text{K/m}$, so it applies to geopotential height
  $z$ (obtained from pressure by the inverse hypsometric relation) rather than to a
  pressure difference in $\text{hPa}$. Applied directly to $(1013.25 - \mathbf{x}_1)$
  it would span only ${\sim}1.5\,\text{K}$ and leave temperature almost pure noise;
  via $z$ it gives $T \approx 275\text{--}291\,\text{K}$.

  $$
  \mathbf{x}_3 \sim \text{Beta}(2, 2) \times 100
  $$

  $$
  u = - \frac{1}{\rho f} \frac{\partial p}{\partial y} + \epsilon_u, \quad v = \frac{1}{\rho f} \frac{\partial p}{\partial x} + \epsilon_v, \quad w = -\frac{RH}{100} \cdot \frac{T - 273.15}{50} + \epsilon_w
  $$

  with $\rho = p / R_d T$, $f = 2\Omega\sin\phi$ at $\phi = 45^\circ\text{N}$, and
  $\epsilon_u, \epsilon_v \sim \mathcal{N}(0, 1.5^2)$, $\epsilon_w \sim \mathcal{N}(0, 0.03^2)$.

  Geostrophic balance needs a pressure *gradient*, which a per-sample scalar $p$
  cannot supply, so `_synoptic_gradients` reconstructs one from the state variables
  using two synoptic regularities — this keeps $\mathbf{y}$ a genuine (noisy)
  function of $\mathbf{x}$, which is what makes the task learnable:

  $$
  -\frac{1}{\rho f}\frac{\partial p}{\partial y} = 5 + 3 d \left(1 + 0.35\,\mathrm{max}(d, 0)\right), \quad
  \frac{1}{\rho f}\frac{\partial p}{\partial x} = 4 \hat{T} \left(1 + 0.6 m\right)
  $$

  where $d = (1013.25 - p)/50$ is the cyclonic depth, $\hat{T} = T - T_{\text{ISA}}(p)$
  the thermal anomaly, and $m = RH/100 - 0.5$ the moisture anomaly. The meridional
  term is the climatological mid-latitude westerly background plus cyclonic
  tightening, steeper for lows than for highs because real cyclones are tighter
  than anticyclones; this asymmetry is what makes the mapping nonlinear. The zonal
  term is warm-sector flow: a warm anomaly at a given pressure sits ahead of a
  trough, where the flow is southerly, and moist air strengthens it.

  Note the sign of $w$: it follows the reanalysis $\omega = dp/dt$ convention, in
  which negative values are ascent. Since $T > 273.15\,\text{K}$ across the sampled
  pressure range, $w$ comes out almost entirely one-signed.

### 2.3. Preprocessing & Cleaning Steps

1. **Filtering & Outliers**: Mask missing or unphysical values ($RH < 0\%$, $T < 180\,\text{K}$). Trim top and bottom $0.5\%$ tail values to mitigate skewness.

2. **Standardization**: Compute feature mean $\mathbf{\mu}_x$ and standard deviation $\mathbf{\sigma}_x$:
   

   $$
   \tilde{x}_{ij} = \frac{x_{ij} - \mu_{xj}}{\sigma_{xj}}
   $$

3. **Target Scaling**: Standardize target output variables $\mathbf{y}_i$ to preserve isotropic variance across $(u, v, w)$.

## 3. Classification Task: Stellar Spectral Classification (Astrophysics)

### 3.1. Physical Problem Formulation

Classifying stellar evolutionary stages based on optical color indices across the Hertzsprung-Russell (HR) diagram.

* **Feature Vector (**$\mathbf{x} \in \mathbb{R}^3$**)**: Photometric optical color indices:

  1. $x_1$: $u - g$ magnitude

  2. $x_2$: $g - r$ magnitude

  3. $x_3$: $r - i$ magnitude

* **Target Vector (**$\mathbf{y} \in \{0, 1\}^3$**)**: One-hot encoded spectral classes:

  * $[1, 0, 0]^T$: *Main Sequence*

  * $[0, 1, 0]^T$: *Red Giant Branch*

  * $[0, 0, 1]^T$: *White Dwarf*

### 3.2. Data Sources & Acquisition

* **Real-World Dataset**: Sloan Digital Sky Survey (SDSS) photometric stellar catalog or Gaia DR3 color data.

* **Synthetic Fallback Generator**: classes are generated as *curved loci* rather than
  isotropic blobs, so the boundaries between them are nonlinear. The main-sequence
  stellar locus is parameterized by $g-r$:

  $$
  (u-g)_{\text{MS}} = 0.78 + 1.95(g-r) - 0.38(g-r)^2, \quad
  (r-i)_{\text{MS}} = 0.33(g-r) + 0.42(g-r)^3
  $$

  * *Main Sequence*: $g-r \sim \text{Uniform}(-0.25, 1.40)$ along the full locus
    (blue A stars through red M dwarfs), scatter $\sigma = 0.06$ per color.

  * *Red Giant Branch*: $g-r = 0.45 + 0.75\,\text{Beta}(2,2)$ — the cool half of the
    locus — offset $+0.18$ in $u-g$ (stronger metal-line blanketing at low surface
    gravity) and bluer in $r-i$, with a larger scatter $\sigma = 0.12$.

  * *White Dwarfs*: a compact blue cluster,
    $u-g \sim \mathcal{N}(0.28, 0.16^2)$, $g-r \sim \mathcal{N}(-0.16, 0.13^2)$,
    $r-i \sim \mathcal{N}(-0.11, 0.10^2)$.

  Separability is deliberately **uneven, and that is the astrophysics**: white dwarfs
  are hot and sit far from everything else in color space, while giants and dwarfs
  overlap, because broadband colors measure temperature and barely constrain surface
  gravity. A trained model therefore reaches near-perfect recall on white dwarfs
  ($\approx 0.99$) and retains visible confusion between the other two
  ($\approx 0.79$ for the main sequence), leaving a cross-entropy floor well above
  zero — which is the point worth showing.

### 3.3. Preprocessing & Cleaning Steps

1. **Extinction Correction**: De-redden raw color indices with SDSS extinction
   coefficients $A_\lambda / E(B-V)$ of $4.239, 3.303, 2.285, 1.698$ for $u, g, r, i$,
   differenced into the three color indices. Applied only on the real-data path, where
   the catalog supplies $E(B-V)$; the synthetic generator produces intrinsic colors.

2. **Color Cuts**: Partition stars using standard color-color selection boundaries on the HR domain.

3. **Class Balancing & Standardization**: Subsample to ensure equal representation ($N=500$ per class) across all three target categories, giving $N=1500$ samples in total, followed by feature Z-score normalization.

## 4. Pipeline Artifact Serialization Schema

The Python preprocessing script (`python/generate_data.py`) outputs a single, structured JSON document at `frontend/public/data/model_runs.json`.

### Units convention

`data.x`, `data.y` and the regression `weights[].outputs` are in **physical units**, so
the corner plot can label real axes. The network trains on **standardized** values, so
`sample_losses` and `total_loss` live in that space; `metadata.standardization` carries
the constants needed to move between the two:
$\tilde{y} = (y - \mu_y) / \sigma_y$. For the classifier, targets are one-hot and
outputs are softmax probabilities, so no de-standardization applies.

### Field reference

* `metadata.loss_name` — `"mse"` or `"cross_entropy"`; selects which definition the
  loss panel renders.
* `metadata.source` — `"synthetic"`, or `"era5"` / `"sdss"` when real catalog files
  were found in `python/raw_data/`. Records which path produced the snapshot.
* `metadata.default_weight_id` — the weight set the UI opens on: the first pretrained
  set (PLAN.md §4B).
* `metadata.quantiles` — per-column $[0.5\%, 99.5\%]$ axis limits. `outputs` is taken
  over **all six weight sets** together, so the $y$ / $\hat{y}$ histograms on the
  bottom-right diagonal can share one axis without clipping a random initialization
  off-screen.
* `weights[].hyperparameters` — the seed for a random initialization; the seed,
  learning rate, batch size and epoch count for a trained one.
* `weights[].sample_losses` — one $L(\hat{\mathbf{y}}_i, \mathbf{y}_i; \Theta)$ per
  sample, in the same row order as `data.x`; `total_loss` is their mean.

Serialized precision is 4 decimals for data and outputs, 6 for losses and parameters.
The whole artifact is ~490 KB and regenerates byte-identically.

### JSON Structure

```json
{
  "datasets": {
    "atmospheric_regression": {
      "metadata": {
        "domain": "Atmospheric Science",
        "task_type": "regression",
        "loss_name": "mse",
        "source": "synthetic",
        "feature_names": ["Pressure (hPa)", "Temperature (K)", "Humidity (%)"],
        "target_names": ["u-wind (m/s)", "v-wind (m/s)", "w-wind (cm/s)"],
        "num_samples": 500,
        "default_weight_id": "trained_opt_1",
        "quantiles": {
          "features": [[804.3463, 1047.6999], [273.996, 292.4474], [7.2294, 92.1109]],
          "targets": [[1.6506, 36.4068], [-19.5978, 19.8119], [-0.3022, 0.0204]],
          "outputs": [[2.7651, 34.2398], [-14.4341, 16.7738], [-0.2719, -0.0154]]
        },
        "standardization": {
          "feature_mean": [922.611208, 282.946097, 49.433633],
          "feature_std": [70.300348, 4.510173, 21.67004],
          "target_mean": [15.990807, 0.080315, -0.098289],
          "target_std": [9.73659, 7.335132, 0.072156]
        }
      },
      "data": {
        "x": [[858.075, 277.156, 81.7277]],
        "y": [[21.4655, -11.0922, -0.0594]]
      },
      "weights": [
        {
          "id": "random_1",
          "name": "Random Init #1",
          "is_trained": false,
          "hyperparameters": {"seed": 0},
          "parameters": {
            "num_params": 387,
            "layers": [
              {"in_features": 3, "out_features": 16, "weights": "...", "biases": "..."},
              {"in_features": 16, "out_features": 16, "weights": "...", "biases": "..."},
              {"in_features": 16, "out_features": 3, "weights": "...", "biases": "..."}
            ]
          },
          "outputs": [[17.3057, 3.1719, -0.0843]],
          "sample_losses": [1.36111, 0.760095],
          "total_loss": 1.019799
        },
        {
          "id": "trained_opt_1",
          "name": "SGD from Init #1 \u2014 lr=0.001, batch 32, 800 epochs",
          "is_trained": true,
          "hyperparameters": {
            "lr": 0.001, "batch_size": 32, "seed": 0,
            "epochs": 800, "optimizer": "SGD"
          },
          "parameters": {"num_params": 387, "layers": []},
          "outputs": [[25.1633, -4.0784, -0.0541]],
          "sample_losses": [0.354632, 0.058982],
          "total_loss": 0.160209
        }
      ]
    },
    "stellar_classification": {
      "metadata": {
        "domain": "Astrophysics",
        "task_type": "classification",
        "loss_name": "cross_entropy",
        "source": "synthetic",
        "feature_names": ["u - g", "g - r", "r - i"],
        "target_names": ["Main Sequence", "Red Giant", "White Dwarf"],
        "num_samples": 1500,
        "default_weight_id": "trained_opt_1",
        "quantiles": {
          "features": [[-0.0393, 2.8316], [-0.4377, 1.3729], [-0.3402, 1.51]],
          "targets": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
          "outputs": [[0.0, 1.0], [0.0, 1.0], [0.0, 0.9994]]
        },
        "standardization": {
          "feature_mean": [1.418916, 0.414599, 0.215468],
          "feature_std": [0.968797, 0.522459, 0.368178],
          "target_mean": [0.0, 0.0, 0.0],
          "target_std": [1.0, 1.0, 1.0]
        }
      },
      "data": {
        "x": [[1.3043, 0.3453, 0.0345]],
        "y": [[1.0, 0.0, 0.0]]
      },
      "weights": ["... six sets, same shape as above; outputs are softmax probabilities ..."]
    }
  }
}
```

## 5. Script Workflow (`python/generate_data.py`)

1. **Environment Setup**: `uv sync`, then `uv run python/generate_data.py` from
   anywhere in the repo — paths resolve from the script, not the working directory.
   `--epochs N` overrides the training length, `--out PATH` the destination. The full
   run takes ~40 s on a laptop CPU and is deterministic: identical bytes every time.

2. **Data Generation/Loading**:

   * Attempt loading real catalog files from `python/raw_data/` —
     `era5_sample.csv` (columns `pressure_hpa`, `temperature_k`,
     `relative_humidity`, `u_ms`, `v_ms`, `w_cms`) and `stellar_colors.csv`
     (columns `u_g`, `g_r`, `r_i`, `class`, optional `ebv`). A NetCDF download
     from the Copernicus CDS must be flattened to that form first, which keeps
     xarray/netCDF4 out of the dependency set.

   * Fallback automatically to synthetic physical generators if raw data files are absent.

3. **Preprocessing**: Apply quantile clipping, feature standardization, target normalization, and one-hot encoding.

4. **PyTorch Execution**:

   * Instantiate model architecture ($3 \to 16 \to 16 \to 3$), $|\Theta| = 387$.

   * Record initial random weights and forward pass predictions, for seeds $0, 1, 2$.

   * Train models using PyTorch SGD — plain, no momentum or schedule — with distinct
     learning rates ($\eta \in \{0.001, 0.01, 0.1\}$) and batch sizes, **starting from
     those same three initializations**: `trained_opt_N` begins at `random_N`, so the two
     are a before/after pair on one optimization rather than unrelated points in
     parameter space. The seed also drives the minibatch shuffle. Note the consequence
     for interpretation: each run differs in initialization *and* learning rate *and*
     batch size, so the spread across the three is not attributable to $\eta$ alone.

   * Record per-sample losses $L(\hat{\mathbf{y}}_i, \mathbf{y}_i; \Theta)$ for each of
     the six exported weight sets. No per-epoch series is stored: learning kinetics are
     outside this tool's scope (PLAN.md §1).

5. **JSON Export**: Serialize features, targets, model outputs, parameter states, and per-sample losses to `frontend/public/data/model_runs.json`.

### Reference values

A clean run produces these total losses, a useful check that the snapshot is sane. The
random sets sit at the uninformed baseline — $\text{MSE} \approx 1$ on standardized
targets, $\text{CE} \approx \ln 3 = 1.0986$ — which is itself worth pointing out in a
lecture.

| Weight set | Starts from | Regression (MSE) | Classification (CE) |
| --- | --- | --- | --- |
| `random_1` / `2` / `3` | seeds 0 / 1 / 2 | 1.0198 / 1.0271 / 0.9717 | 1.1394 / 1.0735 / 1.1073 |
| `trained_opt_1` ($\eta = 0.001$) | `random_1` | 0.1602 | 0.2539 |
| `trained_opt_2` ($\eta = 0.01$) | `random_2` | 0.0761 | 0.1684 |
| `trained_opt_3` ($\eta = 0.1$) | `random_3` | 0.0756 | 0.1470 |

The trained regression loss floors near $0.05$ rather than $0$ because the generator
adds observation noise: the model recovers the conditional mean, not the realization.
