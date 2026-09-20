import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryProvider } from "@/test/queryClientWrapper";
import PredictionsPage from "./page";
import { api } from "@/lib/api";
import type { PaginatedPredictions, ModelVersionSummary } from "@/lib/types";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    listPredictions: vi.fn(),
    listModels: vi.fn(),
  },
}));

const mockPredictions: PaginatedPredictions = {
  items: [
    {
      id: "pred-1",
      request_id: "req-1",
      model_version_id: "model-1",
      prediction: "Yes",
      probability: 0.87,
      created_at: "2026-01-01T12:00:00Z",
    },
  ],
  total: 1,
  limit: 25,
  offset: 0,
};

const mockModels: ModelVersionSummary[] = [
  {
    id: "model-1",
    version_label: "v2",
    algorithm: "random_forest",
    stage: "production",
    metrics: {
      precision: 0.5,
      recall: 0.5,
      f1: 0.5,
      roc_auc: 0.5,
      accuracy: 0.5,
      confusion_matrix: { true_negative: 0, true_positive: 0, false_negative: 0, false_positive: 0 },
    },
    dataset_id: "dataset-1",
    training_job_id: null,
    created_at: "2026-01-01T00:00:00Z",
    promoted_at: null,
  },
];

describe("PredictionsPage", () => {
  beforeEach(() => {
    vi.mocked(api.listPredictions).mockResolvedValue(mockPredictions);
    vi.mocked(api.listModels).mockResolvedValue(mockModels);
  });

  it("renders the prediction history table with real values", async () => {
    render(
      <TestQueryProvider>
        <PredictionsPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("87.00%")).toBeInTheDocument();
    expect(await screen.findByText("v2")).toBeInTheDocument();
  });

  it("does not render raw input feature keys in the list view", async () => {
    render(
      <TestQueryProvider>
        <PredictionsPage />
      </TestQueryProvider>,
    );

    await screen.findByText("Yes");
    expect(screen.queryByText(/input_features/i)).not.toBeInTheDocument();
  });

  it("shows an empty state when there are no predictions", async () => {
    vi.mocked(api.listPredictions).mockResolvedValue({ items: [], total: 0, limit: 25, offset: 0 });

    render(
      <TestQueryProvider>
        <PredictionsPage />
      </TestQueryProvider>,
    );

    expect(await screen.findByText(/no predictions match/i)).toBeInTheDocument();
  });
});
