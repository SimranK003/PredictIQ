import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryProvider } from "@/test/queryClientWrapper";
import ModelDetailPage from "./page";
import { api } from "@/lib/api";
import type { ModelVersionDetail, DatasetDetail, PaginatedPredictions } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "11111111-1111-1111-1111-111111111111" }),
}));

const { MockApiError } = vi.hoisted(() => ({
  MockApiError: class MockApiError extends Error {
    status: number;
    constructor(message: string, status = 404) {
      super(message);
      this.status = status;
    }
  },
}));

vi.mock("@/lib/api", () => ({
  api: {
    getModel: vi.fn(),
    getDataset: vi.fn(),
    listPredictions: vi.fn(),
    getProductionModel: vi.fn(),
  },
  ApiError: MockApiError,
}));

const mockModel: ModelVersionDetail = {
  id: "11111111-1111-1111-1111-111111111111",
  version_label: "v3",
  algorithm: "xgboost",
  stage: "candidate",
  metrics: {
    precision: 0.53,
    recall: 0.68,
    f1: 0.59,
    roc_auc: 0.83,
    accuracy: 0.75,
    confusion_matrix: { true_negative: 10, true_positive: 5, false_negative: 2, false_positive: 3 },
  },
  dataset_id: "dataset-1",
  training_job_id: "job-1",
  created_at: "2026-01-01T00:00:00Z",
  promoted_at: null,
  mlflow_run_id: "run-abc123",
  mlflow_experiment_id: "1",
  artifact_uri: "runs:/run-abc123/model",
  git_commit: "deadbeef",
  params: { n_estimators: 300 },
};

const mockDataset: DatasetDetail = {
  id: "dataset-1",
  filename: "telco_churn.csv",
  schema_name: "telco_customer_churn",
  n_rows: 7043,
  n_columns: 21,
  is_valid: true,
  uploaded_at: "2025-12-01T00:00:00Z",
  content_hash: "abc123hash",
  quality_report: {
    n_rows: 7043,
    n_columns: 21,
    is_valid: true,
    duplicate_row_count: 0,
    column_null_counts: {},
    class_balance: null,
    issues: [],
  },
};

const mockPredictions: PaginatedPredictions = { items: [], total: 42, limit: 1, offset: 0 };

describe("ModelDetailPage", () => {
  beforeEach(() => {
    vi.mocked(api.getModel).mockResolvedValue(mockModel);
    vi.mocked(api.getDataset).mockResolvedValue(mockDataset);
    vi.mocked(api.listPredictions).mockResolvedValue(mockPredictions);
  });

  it("renders model metadata, metrics, and lineage", async () => {
    render(
      <TestQueryProvider>
        <ModelDetailPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText("Model v3")).toBeInTheDocument();
    expect(screen.getAllByText(/run-abc123/).length).toBeGreaterThan(0);
    expect(screen.getByText("deadbeef")).toBeInTheDocument();
    expect(await screen.findByText("abc123hash")).toBeInTheDocument();
    expect(await screen.findByText("telco_churn.csv")).toBeInTheDocument();
    expect(await screen.findByText("42 recorded")).toBeInTheDocument();
  });

  it("shows a Promote button for a candidate model", async () => {
    render(
      <TestQueryProvider>
        <ModelDetailPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByRole("button", { name: /promote to production/i })).toBeInTheDocument();
  });

  it("shows the real backend error message on a 404, not a generic fallback", async () => {
    // Regression test: the page used to check `.isLoading` (only true
    // while actively fetching) instead of `.isPending` (true for the
    // whole pending status, including a retry's backoff pause) before
    // deciding whether to show the loading state — which meant a
    // transient in-between-retries render could fall through to the
    // error branch with error still null, showing a hardcoded fallback
    // string instead of the real backend message. Found via manual
    // browser verification, not by reading the code.
    vi.mocked(api.getModel).mockRejectedValue(new MockApiError("Model version not found.", 404));

    render(
      <TestQueryProvider>
        <ModelDetailPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("Model version not found.");
  });
});
