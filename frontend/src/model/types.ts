/**
 * The shape of frontend/public/data/model_runs.json.
 *
 * This is one half of the contract documented in DATA.md section 4; the other
 * half is python/generate_data.py. Changing either means changing both, plus
 * the doc. Nothing here is computed in the browser that the pipeline already
 * precomputed.
 */

export type DatasetKey = "atmospheric_regression" | "stellar_classification";
export type TaskType = "regression" | "classification";
export type LossName = "mse" | "cross_entropy";

/** Which data path produced the snapshot. The committed one is "synthetic". */
export type DataSource = "synthetic" | "era5" | "sdss";

/** An inclusive [low, high] pair; the corner plot's axis limits. */
export type Range = [number, number];

/** Constants moving between physical units and the space the network trains in. */
export interface Standardization {
  feature_mean: number[];
  feature_std: number[];
  target_mean: number[];
  target_std: number[];
}

export interface DatasetMetadata {
  domain: string;
  task_type: TaskType;
  loss_name: LossName;
  source: DataSource;
  feature_names: string[];
  target_names: string[];
  num_samples: number;
  /** The weight set the UI opens on: the first pretrained set. */
  default_weight_id: string;
  quantiles: {
    features: Range[];
    targets: Range[];
    /** Taken over all six weight sets, so y and yhat can share an axis. */
    outputs: Range[];
  };
  standardization: Standardization;
}

export interface LayerParameters {
  in_features: number;
  out_features: number;
  /** Row-major, [out_features][in_features]. */
  weights: number[][];
  biases: number[];
}

export interface WeightSet {
  id: string;
  name: string;
  is_trained: boolean;
  hyperparameters: Record<string, number | string>;
  parameters: {
    num_params: number;
    layers: LayerParameters[];
  };
  /** Physical units for regression; softmax probabilities for classification. */
  outputs: number[][];
  /** One per sample, in data.x row order, in the standardized training space. */
  sample_losses: number[];
  total_loss: number;
}

export interface Dataset {
  metadata: DatasetMetadata;
  data: {
    /** Features in physical units, [num_samples][3]. */
    x: number[][];
    /** Targets in physical units, or one-hot rows for classification. */
    y: number[][];
  };
  weights: WeightSet[];
}

export interface Artifact {
  datasets: Record<DatasetKey, Dataset>;
}

export const DATASET_KEYS: readonly DatasetKey[] = [
  "atmospheric_regression",
  "stellar_classification",
];

/** Human-readable labels for the dataset selector. */
export const DATASET_LABELS: Record<DatasetKey, string> = {
  atmospheric_regression: "3D Wind Vectors (regression)",
  stellar_classification: "Stellar Spectral Types (classification)",
};

/** What the provenance badge in Panel C says, per PLAN.md section 4C. */
export const SOURCE_LABELS: Record<DataSource, string> = {
  synthetic: "Synthetic — physics-based generator, not observed data",
  era5: "ERA5 reanalysis (ECMWF Copernicus)",
  sdss: "SDSS photometric catalog",
};
