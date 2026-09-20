import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryProvider } from "@/test/queryClientWrapper";
import MonitoringPage from "./page";
import { api } from "@/lib/api";
import type { MonitoringSummary, DriftReport, ModelVersionSummary } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  api: {
    listModels: vi.fn(),
    getMonitoringSummary: vi.fn(),
    getDrift: vi.fn(),
  },
}));

const mockModels: ModelVersionSummary[] = [];

const mockSummary: MonitoringSummary = {
  current_production_model: {
    model_version_id: "model-1",
    version_label: "v1",
    algorithm: "logistic_regression",
    promoted_at: "2026-01-01T00:00:00Z",
  },
  recent_window_hours: 24,
  recent_prediction_count: 5,
  recent_prediction_error_count: 0,
  total_predictions_this_process: 5,
  total_batch_requests_this_process: 0,
  drift_status: "insufficient_data",
  active_training_jobs: 0,
  recent_failed_jobs: [],
  training_job_stats: {
    total_jobs: 1,
    completed_jobs: 1,
    failed_jobs: 0,
    active_jobs: 0,
    average_training_duration_seconds: 4.2,
    n_durations_sampled: 1,
    recent_failures: [],
  },
  model_version_usage: [
    {
      model_version_id: "model-1",
      version_label: "v1",
      algorithm: "logistic_regression",
      stage: "production",
      prediction_count: 5,
      percentage_of_total: 100,
      first_prediction_at: "2026-01-01T00:00:00Z",
      last_prediction_at: "2026-01-01T01:00:00Z",
    },
  ],
};

describe("MonitoringPage", () => {
  beforeEach(() => {
    vi.mocked(api.listModels).mockResolvedValue(mockModels);
    vi.mocked(api.getMonitoringSummary).mockResolvedValue(mockSummary);
  });

  it('renders "insufficient data" honestly rather than a fabricated score', async () => {
    const insufficientDrift: DriftReport = {
      status: "insufficient_data",
      model_version: "v1",
      model_version_id: "model-1",
      window: null,
      sample_size: 5,
      minimum_required: 100,
    };
    vi.mocked(api.getDrift).mockResolvedValue(insufficientDrift);

    render(
      <TestQueryProvider>
        <MonitoringPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText(/insufficient data for drift analysis/i)).toBeInTheDocument();
    expect(screen.getByText(/5 of 100 required predictions/i)).toBeInTheDocument();
  });

  it("renders per-feature drift results with test, score, threshold, and status", async () => {
    const okDrift: DriftReport = {
      status: "ok",
      model_version: "v1",
      model_version_id: "model-1",
      window: null,
      sample_size: 150,
      minimum_required: 100,
      n_sampled_for_computation: 150,
      reference_dataset_id: "dataset-1",
      reference_dataset_content_hash: "abc123",
      reference_training_job_id: null,
      reference_n_rows: 7043,
      drift_detected: true,
      features: [
        {
          feature: "tenure",
          test: "KS",
          score: 0.33,
          threshold: 0.1,
          drift_detected: true,
        },
      ],
    };
    vi.mocked(api.getDrift).mockResolvedValue(okDrift);

    render(
      <TestQueryProvider>
        <MonitoringPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText("tenure")).toBeInTheDocument();
    expect(screen.getByText("KS")).toBeInTheDocument();
    expect(screen.getByText("0.3300")).toBeInTheDocument();
    expect(screen.getAllByText(/drift detected/i).length).toBeGreaterThan(0);
    expect(
      screen.getByText(/does not directly measure model accuracy/i),
    ).toBeInTheDocument();
  });

  it("shows no-production-model state cleanly", async () => {
    vi.mocked(api.getDrift).mockResolvedValue({ status: "no_production_model" });

    render(
      <TestQueryProvider>
        <MonitoringPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText(/no production model available/i)).toBeInTheDocument();
  });
});
