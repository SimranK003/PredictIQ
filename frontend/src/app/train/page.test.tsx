import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TestQueryProvider } from "@/test/queryClientWrapper";
import TrainPage from "./page";
import { api } from "@/lib/api";
import type { DatasetSummary, Job, TrainResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  api: {
    listDatasets: vi.fn(),
    submitTraining: vi.fn(),
    getJob: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}));

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

const mockTrainResponse: TrainResponse = {
  job_id: "job-1",
  dataset_id: "dataset-1",
  status: "queued",
  created_at: "2026-01-01T00:00:00Z",
};

const mockJob: Job = {
  id: "job-1",
  job_type: "train",
  status: "queued",
  dataset_id: "dataset-1",
  n_records: null,
  model_version: null,
  model_version_ids: null,
  result: null,
  error_message: null,
  created_at: "2026-01-01T00:00:00Z",
  started_at: null,
  finished_at: null,
};

describe("TrainPage", () => {
  beforeEach(() => {
    vi.mocked(api.listDatasets).mockResolvedValue(mockDatasets);
    vi.mocked(api.submitTraining).mockResolvedValue(mockTrainResponse);
    vi.mocked(api.getJob).mockResolvedValue(mockJob);
  });

  it("submits a training job with the selected dataset and algorithms, then tracks it", async () => {
    const user = userEvent.setup();
    render(
      <TestQueryProvider>
        <TrainPage />
      </TestQueryProvider>,
    );

    const datasetSelect = await screen.findByLabelText(/1\. dataset/i);
    await user.selectOptions(datasetSelect, "dataset-1");

    const submitButton = screen.getByRole("button", { name: /submit training job/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(api.submitTraining).toHaveBeenCalledWith(
        expect.objectContaining({
          dataset_id: "dataset-1",
          algorithms: ["logistic_regression", "random_forest", "xgboost"],
        }),
      );
    });

    expect(await screen.findByText(/job job-1/i)).toBeInTheDocument();
  });

  it("disables submission until a dataset is selected", async () => {
    render(
      <TestQueryProvider>
        <TrainPage />
      </TestQueryProvider>,
    );

    const submitButton = await screen.findByRole("button", { name: /submit training job/i });
    expect(submitButton).toBeDisabled();
  });
});
