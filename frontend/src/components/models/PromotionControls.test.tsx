import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryProvider } from "@/test/queryClientWrapper";
import { PromotionControls } from "./PromotionControls";
import type { ModelVersionDetail } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  api: { promoteModel: vi.fn(), rollbackModel: vi.fn(), getProductionModel: vi.fn() },
  ApiError: class MockApiError extends Error {
    status: number;
    constructor(message: string, status = 400) {
      super(message);
      this.status = status;
    }
  },
}));

const CANDIDATE: ModelVersionDetail = {
  id: "m1",
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
  params: {},
};

describe("PromotionControls", () => {
  it("shows the Promote button for an admin viewing a candidate", () => {
    render(
      <TestQueryProvider>
        <PromotionControls model={CANDIDATE} />
      </TestQueryProvider>,
    );
    expect(screen.getByRole("button", { name: /promote to production/i })).toBeInTheDocument();
  });

  it("renders nothing for a non-admin user, even for a promotable candidate", () => {
    render(
      <TestQueryProvider user={{ ...adminlessUser() }}>
        <PromotionControls model={CANDIDATE} />
      </TestQueryProvider>,
    );
    expect(screen.queryByRole("button", { name: /promote to production/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /roll back/i })).not.toBeInTheDocument();
  });

  it("renders nothing for an unauthenticated (null) user", () => {
    render(
      <TestQueryProvider user={null}>
        <PromotionControls model={CANDIDATE} />
      </TestQueryProvider>,
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});

function adminlessUser() {
  return {
    id: "u2",
    email: "member@test.local",
    is_admin: false,
    created_at: "2026-01-01T00:00:00Z",
    last_login_at: null,
  };
}
