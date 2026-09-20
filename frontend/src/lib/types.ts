/**
 * TypeScript types mirroring the FastAPI backend's actual response
 * shapes (backend/app/schemas/*.py). Kept in this one file so a backend
 * schema change has one obvious place to update on the frontend too.
 */

export type ModelStage = "candidate" | "production" | "previous" | "archived";
export type JobType = "ingest" | "train" | "batch_predict";
export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface ConfusionMatrix {
  true_negative: number;
  true_positive: number;
  false_negative: number;
  false_positive: number;
}

export interface ModelMetrics {
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  accuracy: number;
  confusion_matrix: ConfusionMatrix;
}

export interface ModelVersionSummary {
  id: string;
  version_label: string;
  algorithm: string;
  stage: ModelStage;
  metrics: ModelMetrics;
  dataset_id: string;
  training_job_id: string | null;
  created_at: string;
  promoted_at: string | null;
}

export interface ModelVersionDetail extends ModelVersionSummary {
  mlflow_run_id: string;
  mlflow_experiment_id: string;
  artifact_uri: string;
  git_commit: string;
  params: Record<string, unknown>;
}

export interface DatasetSummary {
  id: string;
  filename: string;
  schema_name: string;
  n_rows: number;
  is_valid: boolean;
  uploaded_at: string;
}

export interface ValidationIssue {
  check: string;
  severity: "error" | "warning";
  message: string;
  details: Record<string, unknown>;
}

export interface DataQualityReport {
  n_rows: number;
  n_columns: number;
  is_valid: boolean;
  duplicate_row_count: number;
  column_null_counts: Record<string, number>;
  class_balance: Record<string, number> | null;
  issues: ValidationIssue[];
}

export interface DatasetDetail extends DatasetSummary {
  content_hash: string;
  n_columns: number;
  quality_report: DataQualityReport;
}

export interface JobResult {
  // Training jobs
  best_algorithm?: string;
  best_test_roc_auc?: number;
  best_model_version_id?: string;
  algorithms_trained?: string[];
  candidate_model_version_ids?: string[];
  partial_candidate_model_version_ids?: string[];
  // Batch prediction jobs
  n_processed?: number;
  prediction_ids?: string[];
}

export interface Job {
  id: string;
  job_type: JobType;
  status: JobStatus;
  dataset_id: string | null;
  n_records: number | null;
  model_version: string | null;
  model_version_ids: string[] | null;
  result: JobResult | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface PaginatedJobs {
  items: Job[];
  total: number;
  limit: number;
  offset: number;
}

export interface PredictionSummary {
  id: string;
  request_id: string;
  model_version_id: string;
  prediction: string;
  probability: number;
  created_at: string;
}

export interface PredictionDetail {
  id: string;
  request_id: string;
  model_version_id: string;
  model_version_label: string;
  algorithm: string;
  input_features: Record<string, unknown>;
  prediction: string;
  probability: number;
  latency_ms: number;
  created_at: string;
}

export interface PaginatedPredictions {
  items: PredictionSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface AuthUser {
  id: string;
  email: string;
  is_admin: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface HealthStatus {
  status: "ok" | "degraded" | "unhealthy";
  database: "ok" | "unavailable";
  redis: "ok" | "unavailable";
  production_model: {
    available: boolean;
    version: string | null;
    algorithm: string | null;
  };
}

export interface ModelUsage {
  model_version_id: string;
  version_label: string | null;
  algorithm: string | null;
  stage: ModelStage | null;
  prediction_count: number;
  percentage_of_total: number;
  first_prediction_at: string;
  last_prediction_at: string;
}

export interface TrainingJobFailureSummary {
  job_id: string;
  dataset_id: string | null;
  error_message: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface TrainingJobStats {
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  active_jobs: number;
  average_training_duration_seconds: number | null;
  n_durations_sampled: number;
  recent_failures: TrainingJobFailureSummary[];
}

export interface RecentFailedJob {
  job_id: string;
  job_type: JobType;
  error_message: string | null;
  created_at: string;
}

export interface MonitoringSummary {
  current_production_model: {
    model_version_id: string;
    version_label: string;
    algorithm: string;
    promoted_at: string | null;
  } | null;
  recent_window_hours: number;
  recent_prediction_count: number;
  recent_prediction_error_count: number;
  /** Prometheus counter scoped to THIS API process only — see backend
   * monitoring/metrics.py. Never present this as a global lifetime total. */
  total_predictions_this_process: number;
  total_batch_requests_this_process: number;
  drift_status: string;
  active_training_jobs: number;
  recent_failed_jobs: RecentFailedJob[];
  training_job_stats: TrainingJobStats;
  model_version_usage: ModelUsage[];
}

export type DriftTestKind = "KS" | "PSI";

export interface DriftFeatureResult {
  feature: string;
  test: DriftTestKind;
  score: number | null;
  threshold: number;
  drift_detected: boolean;
  p_value?: number;
  reference_n?: number;
  production_n?: number;
  critical_threshold?: number;
  severity?: "none" | "warning" | "critical";
  note?: string;
}

export type DriftReport =
  | { status: "no_production_model" }
  | {
      status: "insufficient_data";
      model_version: string;
      model_version_id: string;
      window: string | null;
      sample_size: number;
      minimum_required: number;
    }
  | {
      status: "ok";
      model_version: string;
      model_version_id: string;
      window: string | null;
      sample_size: number;
      minimum_required: number;
      n_sampled_for_computation: number;
      reference_dataset_id: string;
      reference_dataset_content_hash: string;
      reference_training_job_id: string | null;
      reference_n_rows: number;
      drift_detected: boolean;
      features: DriftFeatureResult[];
    };

export interface TrainRequest {
  dataset_id: string;
  algorithms?: string[] | null;
  test_size?: number;
  val_size?: number;
  random_state?: number;
  logistic_regression_c?: number;
  logistic_regression_max_iter?: number;
  random_forest_n_estimators?: number;
  random_forest_max_depth?: number;
  random_forest_min_samples_leaf?: number;
  xgboost_n_estimators?: number;
  xgboost_max_depth?: number;
  xgboost_learning_rate?: number;
  xgboost_subsample?: number;
}

export interface TrainResponse {
  job_id: string;
  dataset_id: string;
  status: JobStatus;
  created_at: string;
}

export const ALGORITHMS = ["logistic_regression", "random_forest", "xgboost"] as const;
export type Algorithm = (typeof ALGORITHMS)[number];
