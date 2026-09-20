"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { Card } from "@/components/ui/Card";
import { ErrorState, LoadingState } from "@/components/ui/States";
import { ALGORITHMS, type Algorithm, type TrainRequest } from "@/lib/types";
import { TrainingJobTracker } from "@/components/train/TrainingJobTracker";

const ALGORITHM_LABELS: Record<Algorithm, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
};

export default function TrainPage() {
  const datasetsQuery = useQuery({ queryKey: queryKeys.datasets, queryFn: api.listDatasets });

  const [datasetId, setDatasetId] = useState("");
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<Algorithm[]>([...ALGORITHMS]);
  const [testSize, setTestSize] = useState(0.15);
  const [valSize, setValSize] = useState(0.15);
  const [randomState, setRandomState] = useState(42);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [xgboostNEstimators, setXgboostNEstimators] = useState(300);
  const [randomForestNEstimators, setRandomForestNEstimators] = useState(300);
  const [submittedJobId, setSubmittedJobId] = useState<string | null>(null);

  const submitMutation = useMutation({
    mutationFn: (payload: TrainRequest) => api.submitTraining(payload),
    onSuccess: (data) => setSubmittedJobId(data.job_id),
  });

  const validDatasets = datasetsQuery.data?.filter((d) => d.is_valid) ?? [];

  const toggleAlgorithm = (algorithm: Algorithm) => {
    setSelectedAlgorithms((prev) =>
      prev.includes(algorithm) ? prev.filter((a) => a !== algorithm) : [...prev, algorithm],
    );
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!datasetId || selectedAlgorithms.length === 0) return;

    const payload: TrainRequest = {
      dataset_id: datasetId,
      algorithms: selectedAlgorithms,
      test_size: testSize,
      val_size: valSize,
      random_state: randomState,
    };
    if (selectedAlgorithms.includes("xgboost")) {
      payload.xgboost_n_estimators = xgboostNEstimators;
    }
    if (selectedAlgorithms.includes("random_forest")) {
      payload.random_forest_n_estimators = randomForestNEstimators;
    }
    submitMutation.mutate(payload);
  };

  return (
    <PageShell
      title="Train a Model"
      description="Submit an asynchronous training job — this returns immediately with a job ID; training runs in a Celery worker."
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="New training job">
          {datasetsQuery.isPending && <LoadingState label="Loading datasets…" />}
          {datasetsQuery.error && (
            <ErrorState
              message={
                datasetsQuery.error instanceof Error ? datasetsQuery.error.message : "Failed to load datasets."
              }
              onRetry={() => datasetsQuery.refetch()}
            />
          )}

          {datasetsQuery.data && (
            <form onSubmit={handleSubmit} className="flex flex-col gap-5">
              <div>
                <label htmlFor="dataset" className="block text-sm font-medium text-slate-700">
                  1. Dataset
                </label>
                {validDatasets.length === 0 ? (
                  <p className="mt-1 text-sm text-slate-500">
                    No valid datasets available. Upload one via <code>POST /datasets</code> first.
                  </p>
                ) : (
                  <select
                    id="dataset"
                    required
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    className="mt-1 block w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700"
                  >
                    <option value="">Select a dataset…</option>
                    {validDatasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.filename} ({d.n_rows.toLocaleString()} rows)
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <fieldset>
                <legend className="text-sm font-medium text-slate-700">2. Algorithms</legend>
                <div className="mt-2 flex flex-col gap-2">
                  {ALGORITHMS.map((algorithm) => (
                    <label key={algorithm} className="flex items-center gap-2 text-sm text-slate-700">
                      <input
                        type="checkbox"
                        checked={selectedAlgorithms.includes(algorithm)}
                        onChange={() => toggleAlgorithm(algorithm)}
                        className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      {ALGORITHM_LABELS[algorithm]}
                    </label>
                  ))}
                </div>
                {selectedAlgorithms.length === 0 && (
                  <p className="mt-1 text-xs text-red-600">Select at least one algorithm.</p>
                )}
              </fieldset>

              <div>
                <button
                  type="button"
                  onClick={() => setShowAdvanced((v) => !v)}
                  className="text-sm font-medium text-indigo-600 hover:underline"
                  aria-expanded={showAdvanced}
                >
                  3. {showAdvanced ? "Hide" : "Configure"} hyperparameters
                </button>
                {showAdvanced && (
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <label className="flex flex-col text-xs text-slate-500">
                      Test size
                      <input
                        type="number"
                        step="0.01"
                        min="0.05"
                        max="0.5"
                        value={testSize}
                        onChange={(e) => setTestSize(Number(e.target.value))}
                        className="mt-1 rounded-md border border-slate-200 px-2 py-1 text-sm"
                      />
                    </label>
                    <label className="flex flex-col text-xs text-slate-500">
                      Validation size
                      <input
                        type="number"
                        step="0.01"
                        min="0.05"
                        max="0.5"
                        value={valSize}
                        onChange={(e) => setValSize(Number(e.target.value))}
                        className="mt-1 rounded-md border border-slate-200 px-2 py-1 text-sm"
                      />
                    </label>
                    <label className="flex flex-col text-xs text-slate-500">
                      Random seed
                      <input
                        type="number"
                        value={randomState}
                        onChange={(e) => setRandomState(Number(e.target.value))}
                        className="mt-1 rounded-md border border-slate-200 px-2 py-1 text-sm"
                      />
                    </label>
                    {selectedAlgorithms.includes("random_forest") && (
                      <label className="flex flex-col text-xs text-slate-500">
                        Random forest trees
                        <input
                          type="number"
                          min="10"
                          max="1000"
                          value={randomForestNEstimators}
                          onChange={(e) => setRandomForestNEstimators(Number(e.target.value))}
                          className="mt-1 rounded-md border border-slate-200 px-2 py-1 text-sm"
                        />
                      </label>
                    )}
                    {selectedAlgorithms.includes("xgboost") && (
                      <label className="flex flex-col text-xs text-slate-500">
                        XGBoost trees
                        <input
                          type="number"
                          min="10"
                          max="1000"
                          value={xgboostNEstimators}
                          onChange={(e) => setXgboostNEstimators(Number(e.target.value))}
                          className="mt-1 rounded-md border border-slate-200 px-2 py-1 text-sm"
                        />
                      </label>
                    )}
                  </div>
                )}
              </div>

              <button
                type="submit"
                disabled={
                  !datasetId || selectedAlgorithms.length === 0 || submitMutation.isPending
                }
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {submitMutation.isPending ? "Submitting…" : "4. Submit training job"}
              </button>

              {submitMutation.isError && (
                <p className="text-sm text-red-600">
                  {submitMutation.error instanceof ApiError
                    ? submitMutation.error.message
                    : "Failed to submit the training job."}
                </p>
              )}
            </form>
          )}
        </Card>

        <div>
          {submittedJobId ? (
            <TrainingJobTracker jobId={submittedJobId} />
          ) : (
            <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-300 p-8 text-center text-sm text-slate-400">
              Submit a job to see its progress here — queued → running → completed/failed.
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
