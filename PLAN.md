# PLAN.md: Loss Functions and Optimization Interactive Visualizer

## 1. Project Overview & Objectives
This project is an interactive, web-based educational tool designed primarily for instructor-led presentations in an "AI for Science" curriculum, with secondary support for self-guided student exploration.

### Core Educational Goals
- Build intuition for the relationship between multidimensional scientific data ($\mathbf{x}$) and model predictions ($\hat{\mathbf{y}}$).
- Demonstrate how model parameters $\Theta$ govern predictions and how prediction-target discrepancies map to per-sample loss values $L(\hat{\mathbf{y}}_i, \mathbf{y}_i; \Theta)$.
- Contrast regression and classification tasks using identical functional flow and layout structures.

### Design Principles
- **Instructor-Controlled Progression**: Panel layout, visibility, and collapse states are manually managed by the instructor using modular drag-and-drop tiles (`Gridstack.js`).
- **Model-Agnostic Paradigm**: Focus strictly on the training problem mapping $\mathbf{x} \to f(\mathbf{x};\Theta) \to \hat{\mathbf{y}} \leftrightarrow \mathbf{y} \to L$. Optimization steps, validation/testing splits, and learning kinetics are excluded from this tool.
- **Strict Visual Coding**: Distinct color schemes distinguish dataset-fixed components from parameter-dependent components.

---

## 2. Technology Stack & Environment

### Data Generation Pipeline (Python)
- **Package Manager**: `uv`
- **Machine Learning Core**: PyTorch
- **Output Format**: Single static JSON file (`frontend/public/data/model_runs.json`) containing dataset feature arrays, target arrays, pre-computed model parameters, outputs, and per-sample losses.

### Web Application (Frontend)
- **Build Tool / Server**: Vite
- **Language**: TypeScript (`.ts`)
- **Layout Manager**: `Gridstack.js` (for draggable, collapsible, and resizable layout panels)
- **Visualization Engines**: D3.js, SVG, and `nnet-svg` (`v0.1.1` from npm for neural network architecture diagrams)
- **LaTeX Math Rendering**: KaTeX or MathJax (for rendered loss equations)
- **Hosting / CI**: GitHub Pages deployed automatically via GitHub Actions from `https://github.com/dkirkby/data-model-loss`

---

## 3. Data Specs & Model Architecture

### Scientific Datasets (3 Input Features $\to$ 3 Output Targets)

1. **Regression Task: 3D Wind Vector Prediction (Atmospheric Science & Fluid Dynamics)**

   * Features $\mathbf{x} \in \mathbb{R}^3$: Atmospheric Pressure $p$ ($\text{hPa}$), Air Temperature $T$ ($\text{K}$), Relative Humidity $RH$ ($\%$).

   * Targets $\mathbf{y} \in \mathbb{R}^3$: Zonal Wind $u$ ($\text{m/s}$), Meridional Wind $v$ ($\text{m/s}$), Vertical Velocity $w$ ($\text{cm/s}$).

   * Loss Function: Mean Squared Error (MSE), $L_{\text{MSE}} = \frac{1}{3}\sum_{l=1}^3 (\hat{y}_l - y_l)^2$.

2. **Classification Task: Stellar Spectral Classification (Astrophysics)**

   * Features $\mathbf{x} \in \mathbb{R}^3$: Optical color indices $u-g$, $g-r$, $r-i$ (photometric magnitudes).

   * Targets $\mathbf{y} \in \{0, 1\}^3$: One-hot encoded stellar spectral types (*Main Sequence*, *Red Giant Branch*, *White Dwarf*).

   * Loss Function: Categorical Cross-Entropy (CE), $L_{\text{CE}} = -\sum_{l=1}^3 y_l \log(\hat{y}_l)$ with Softmax output activation.

See DATA.md for details.

### Neural Network Model Architecture
- **Structure**: Fixed dense feedforward neural network with 2 hidden layers ($3 \to 16 \to 16 \to 3$).
- **Activations**: ReLU (or Tanh) for hidden layers; Identity for regression output, Softmax for classification output.
- **Weight Configurations per Dataset**: 6 total sets of weights $\Theta$, in 3 pairs:
  - 3 sets from random initialization (seeds $0, 1, 2$).
  - 3 sets from SGD runs started **from those same three initializations**, with
    different learning rates and batch sizes. `trained_opt_N` begins at `random_N`, so
    selecting one then the other shows before and after on a single optimization.

---

## 4. UI Layout & Component Architecture

### Layout System (`Gridstack.js`)
The workspace is organized into resizable, draggable, and collapsible grid panels:
1. **Top Container**: Data Flow Chart SVG Panel.
2. **Main Workspace**:
   - **Panel A**: 6x6 Multi-Block Corner Scatter & Histogram Plot.
   - **Panel B**: Model Architecture Diagram (`nnet-svg`) & Weight Selector.
   - **Panel C**: Dataset Description & Inspector Panel.
   - **Panel D**: Loss Function Definition Panel (LaTeX & Text).

---

### Detailed Component Specifications

#### Top Container. Central Data Flow Chart (SVG Panel)
Displays the structural pipeline from left to right:
- **Blocks**:
  1. `FEATURES` ($x_{ij}$): Tall rectangle with outline `#fecc07` (Dataset component). Height represents samples $i=1\dots N$.
  2. `MODEL` ($f_l(\mathbf{x}_i;\Theta)$): Rounded rectangle with background `#255799` and white text.
  3. `OUTPUT` ($\hat{y}_{il}$): Tall rectangle with background `#255799` and white text.
  4. `TARGET`: ($y_{il}$): Tall rectangle with outline `#fecc07` (Dataset component).
- **Labels**:
  - Top labels: `FEATURES`, `MODEL`, `OUTPUT`, `TARGET`.
  - Centered math labels: $x_{ij}$, $f_l(\mathbf{x}_i;\Theta)$, $\hat{y}_{il}$, $y_{il}$.
- **Connections & Arrows**:
  - Thin left-to-right arrows connecting sample elements from `FEATURES` $\to$ `MODEL` $\to$ `OUTPUT`.
  - Dark green bidirectional vertical lines connecting aligned elements of `OUTPUT` and `TARGET`.
  - 90-degree rotated label along green lines: `LOSS FUNCTION $L(\hat{\mathbf{y}},\mathbf{y};\Theta)$`.

#### A. 6x6 Corner Plot Panel
Arranged as a $6 \times 6$ matrix of subplots with axes omitted and data scaled to $[0.5\%, 99.5\%]$ quantiles:
- **Diagonal Subplots (1 to 6)**:
  - Subplots 1–3: Histograms of input data features $x_1, x_2, x_3$.
  - Subplots 4–6: Combined histograms comparing target values $y_1, y_2, y_3$ (outlined in `#fecc07`) and model predictions $\hat{y}_1, \hat{y}_2, \hat{y}_3$ (filled with `#255799`).
- **Top-Left $3 \times 3$ Block (Feature Space)**:
  - Outlined in `#fecc07`.
  - Upper & lower off-diagonals show 2D scatter plots of data features $x_i$ vs $x_j$.
- **Bottom-Right $3 \times 3$ Block (Output / Target Space)**:
  - Lower off-diagonal: 2D scatter plots of target values $y_i$ vs $y_j$.
  - Upper off-diagonal: 2D scatter plots of model outputs $\hat{y}_i$ vs $\hat{y}_j$.
- **Bottom-Left $3 \times 3$ Block (Feature vs Target Space)**:
  - 2D scatter plots of features $x_i$ vs target values $y_j$.
- **Top-Right $3 \times 3$ Block (Feature vs Output Space)**:
  - 2D scatter plots of features $x_i$ vs model output values $\hat{y}_j$.
- **Controls & Interactions**:
  - Toggle Checkbox: *"Points colored by loss value"*. When checked, point colors transition along a color ramp according to per-sample loss $L_i$ for the selected weight set $\Theta$.
  - Visual Highlighting: Clicking `FEATURES` or `TARGET` or `OUTPUT` in the flow chart temporarily flashes/highlights the corresponding subplots. Clicking the loss function area temporarily flashes/highlights the histograms along the bottom-right 3x3 diagonal. Clicking `MODEL` in the flowchart temporarily flashes/highlights subplots involving yhat.

#### B. Model Architecture Panel (`nnet-svg`)
- Renders the neural network layers visually using `nnet-svg` (`v0.1.1`).
- Edges labeled with weights $w_{ijk}$ and nodes labeled with biases $b_{jk}$.
- Badge displaying **Total Parameter Count**: $|\Theta|$.
- Weight Selection Controls: Radio buttons or dropdown to choose between 3 Random Initializations and 3 Pretrained Weight sets (Default: 1st Pretrained Set).

#### C. Dataset Description & Inspector Panel
- **Dataset selector**: switches between the regression and classification datasets.
  §4's panel list does not assign this control a home; it lives here because this is the
  panel that describes the dataset. Switching resets the weight selection to the new
  dataset's `default_weight_id`.
- **Header**: `metadata.domain` and the task type, with the sample count $N$.
- **Variable list**: `feature_names` and `target_names` with their units, in the column
  order used everywhere else, so a corner-plot subplot can be read back to a variable.
- **Provenance badge** (`metadata.source`): states plainly whether the active dataset is
  **synthetic** — generated by the physics-based fallback in `DATA.md` §2.2 / §3.2 — or
  real (`era5`, `sdss`). The committed snapshot is synthetic, and a student must never
  mistake it for reanalysis or a photometric catalog. This is the one element of the
  panel that is not optional.
- **Inspector**: on selecting a sample, show its $\mathbf{x}_i$, $\mathbf{y}_i$,
  $\hat{\mathbf{y}}_i$ for the active $\Theta$, and its per-sample loss $L_i$.

#### D. Loss Function Definition Panel
- Displays mathematical equations (rendered in LaTeX via KaTeX/MathJax) and explanatory text for:
  - Mean Squared Error (MSE) when Regression dataset is active.
  - Categorical Cross-Entropy (CE) when Classification dataset is active.


---

## 5. Implementation Steps

### Phase 1: Data & Model Training Pipeline (`python/`)
1. Setup Python environment using `uv init` and add `torch`, `numpy`, `scipy`.
2. Write script `generate_data.py` to fetch/synthesize atmospheric data.
3. Build model training script to train dense $3 \to 16 \to 16 \to 3$ neural network using SGD under 3 hyperparameter configurations.
4. Export evaluation results for all 6 weight sets (3 random + 3 SGD checkpoints) into `frontend/public/data/model_runs.json`.

### Phase 2: Web Application Setup (`frontend/`)
1. Initialize Vite project with TypeScript template: `npm create vite@latest . -- --template vanilla-ts`.
2. Install dependencies: `gridstack`, `d3`, `nnet-svg`, `katex`.
3. Configure `Gridstack.js` responsive layout container in `index.html`.

### Phase 3: Visual Components Development
1. Build `Flowchart.ts` with SVG element mapping and click handlers.
2. Build `CornerPlot.ts` with D3.js matrix renderers, quantile scaling $[0.5\%, 99.5\%]$, and loss color mapping.
3. Integrate `nnet-svg` in `ModelPanel.ts` with parameterized weight controls.
4. Build `LossPanel.ts` with dynamic KaTeX formula rendering.

### Phase 4: CI/CD & Deployment
1. Create `.github/workflows/deploy.yml` for automated GitHub Pages deployment on push to `main`.

---

## 6. Manual QA Checklist for User Verification

Do not perform automated browser tests. Instead, the following manual test checklist must be executed by the user:

- [ ] **Flowchart Rendering**: Confirm block colors match specification (`#fecc07` for features/targets, `#255799` for model/outputs) and math labels render correctly.
- [ ] **Interactivity**: Click each flowchart block (`FEATURES`, `MODEL`, `OUTPUT`, `TARGET`, Loss region) and verify corresponding panels open and highlight correctly.
- [ ] **Gridstack Panels**: Verify panels can be dragged, resized, and collapsed without breaking SVG visual element aspect ratios.
- [ ] **Corner Plot Quantiles**: Check that scatter plots and histograms do not display extreme outlier clipping issues (limits set to 0.5%–99.5% quantiles).
- [ ] **Loss Coloring**: Toggle "points colored by loss value" and verify scatter point colors adjust smoothly according to per-sample loss magnitudes.
- [ ] **Weight Switching**: Switch weight selection between Random 1–3 and Pretrained 1–3, checking that output histograms and loss colors update instantly.
- [ ] **Dataset Switching**: Switch between Regression (3D Wind Vectors) and Classification (Stellar Spectral Types); check that loss equations toggle between MSE and Cross-Entropy.
