import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryProvider } from "@/test/queryClientWrapper";
import ModelsPage from "./page";
import { api } from "@/lib/api";
import type { ModelVersionSummary, DatasetSummary } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  api: {
    listModels: vi.fn(),
    listDatasets: vi.fn(),
  },
}));

const mockModels: ModelVersionSummary[] = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    version_label: "v1",
    algorithm: "logistic_regression",
    stage: "production",
    metrics: {
      precision: 0.51,
      recall: 0.81,
      f1: 0.62,
      roc_auc: 0.85,
      accuracy: 0.74,
      confusion_matrix: { true_negative: 1, true_positive: 1, false_negative: 1, false_positive: 1 },
    },
    dataset_id: "dataset-1",
    training_job_id: "job-1",
    created_at: "2026-01-01T00:00:00Z",
    promoted_at: "2026-01-02T00:00:00Z",
  },
];

const mockDatasets: DatasetSummary[] = [
  {
    id: "dataset-1",
    filename: "telco_churn.csv",
    schema_name: "telco_customer_churn",
    n_rows: 7043,
    is_valid: true,
    uploaded_at: "2025-12-01T00:00:00Z",
  },
];

describe("ModelsPage", () => {
  beforeEach(() => {
    vi.mocked(api.listModels).mockResolvedValue(mockModels);
    vi.mocked(api.listDatasets).mockResolvedValue(mockDatasets);
  });

  it("renders model versions with metrics and dataset filename", async () => {
    render(
      <TestQueryProvider>
        <ModelsPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText("v1")).toBeInTheDocument();
    expect(screen.getByText("logistic_regression")).toBeInTheDocument();
    expect(screen.getByText("production")).toBeInTheDocument();
    expect(screen.getByText("85.0%")).toBeInTheDocument(); // ROC-AUC
    expect(await screen.findByText("telco_churn.csv")).toBeInTheDocument();
  });

  it("shows an empty state when no models exist", async () => {
    vi.mocked(api.listModels).mockResolvedValue([]);

    render(
      <TestQueryProvider>
        <ModelsPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText(/no model versions match/i)).toBeInTheDocument();
  });
});
