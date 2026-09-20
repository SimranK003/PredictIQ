"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { Card, MetricTile } from "@/components/ui/Card";
import { ErrorState, LoadingState } from "@/components/ui/States";
import { StageBadge } from "@/components/ui/StatusBadge";
import { formatDateTime, formatPercent, truncateId } from "@/lib/format";
import { PromotionControls } from "@/components/models/PromotionControls";
import { buildLineageSteps, LineageDiagram } from "@/components/models/LineageDiagram";

export default function ModelDetailPage() {
  const params = useParams<{ id: string }>();
  const modelId = params.id;

  const modelQuery = useQuery({
    queryKey: queryKeys.model(modelId),
    queryFn: () => api.getModel(modelId),
  });
  const datasetQuery = useQuery({
    queryKey: queryKeys.dataset(modelQuery.data?.dataset_id ?? ""),
    queryFn: () => api.getDataset(modelQuery.data!.dataset_id),
    enabled: Boolean(modelQuery.data?.dataset_id),
  });
  const predictionsCountQuery = useQuery({
    queryKey: queryKeys.predictions({ modelVersionId: modelId, limit: 1 }),
    queryFn: () => api.listPredictions({ modelVersionId: modelId, limit: 1 }),
    enabled: Boolean(modelQuery.data),
  });

  if (modelQuery.isPending) {
    return (
      <PageShell title="Model details">
        <LoadingState label="Loading model…" />
      </PageShell>
    );
  }

  if (modelQuery.error || !modelQuery.data) {
    return (
      <PageShell title="Model details">
        <ErrorState
          message={
            modelQuery.error instanceof Error ? modelQuery.error.message : "Model not found."
          }
          onRetry={() => modelQuery.refetch()}
        />
      </PageShell>
    );
  }

  const model = modelQuery.data;
  const metrics = model.metrics;

  return (
    <PageShell
      title={`Model ${model.version_label}`}
      description={`${model.algorithm} · registered ${formatDateTime(model.created_at)}`}
      actions={<PromotionControls model={model} />}
    >
      <div className="flex flex-col gap-6">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <MetricTile label="ROC-AUC" value={formatPercent(metrics.roc_auc)} />
          <MetricTile label="Precision" value={formatPercent(metrics.precision)} />
          <MetricTile label="Recall" value={formatPercent(metrics.recall)} />
          <MetricTile label="F1" value={formatPercent(metrics.f1)} />
          <MetricTile label="Accuracy" value={formatPercent(metrics.accuracy)} />
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card title="Model metadata">
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <div>
                <dt className="text-xs text-slate-500">Stage</dt>
                <dd className="mt-1">
                  <StageBadge stage={model.stage} />
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Algorithm</dt>
                <dd className="mt-1 text-slate-800">{model.algorithm}</dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-slate-500">MLflow run</dt>
                <dd className="mt-1 break-all font-mono text-xs text-slate-800">
                  {model.mlflow_run_id}
                </dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-slate-500">Git commit</dt>
                <dd className="mt-1 break-all font-mono text-xs text-slate-800">
                  {model.git_commit}
                </dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-slate-500">Dataset ID</dt>
                <dd className="mt-1 break-all font-mono text-xs text-slate-800">
                  {model.dataset_id}
                </dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-slate-500">Dataset content hash (SHA-256)</dt>
                <dd className="mt-1 break-all font-mono text-xs text-slate-800">
                  {datasetQuery.data?.content_hash ?? "Loading…"}
                </dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-slate-500">Training job</dt>
                <dd className="mt-1 font-mono text-xs text-slate-800">
                  {model.training_job_id ? truncateId(model.training_job_id, 16) : "None (CLI-trained)"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Created</dt>
                <dd className="mt-1 text-slate-800">{formatDateTime(model.created_at)}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Promoted</dt>
                <dd className="mt-1 text-slate-800">{formatDateTime(model.promoted_at)}</dd>
              </div>
            </dl>
          </Card>

          <Card title="Lineage">
            <LineageDiagram
              steps={buildLineageSteps({
                datasetFilename: datasetQuery.data?.filename ?? truncateId(model.dataset_id),
                datasetId: model.dataset_id,
                trainingJobId: model.training_job_id,
                mlflowRunId: model.mlflow_run_id,
                modelVersionLabel: model.version_label,
                modelVersionId: model.id,
                predictionCount: predictionsCountQuery.data?.total ?? 0,
              })}
            />
          </Card>
        </div>

        <Card title="Confusion matrix (test set)">
          <div className="grid grid-cols-2 gap-3 sm:max-w-md">
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-center">
              <p className="text-xs text-slate-500">True negative</p>
              <p className="text-lg font-semibold text-slate-800">
                {metrics.confusion_matrix.true_negative}
              </p>
            </div>
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-center">
              <p className="text-xs text-slate-500">False positive</p>
              <p className="text-lg font-semibold text-slate-800">
                {metrics.confusion_matrix.false_positive}
              </p>
            </div>
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-center">
              <p className="text-xs text-slate-500">False negative</p>
              <p className="text-lg font-semibold text-slate-800">
                {metrics.confusion_matrix.false_negative}
              </p>
            </div>
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 text-center">
              <p className="text-xs text-slate-500">True positive</p>
              <p className="text-lg font-semibold text-slate-800">
                {metrics.confusion_matrix.true_positive}
              </p>
            </div>
          </div>
        </Card>

        <Card title="Training configuration">
          <pre className="overflow-x-auto rounded-md bg-slate-900 p-3 text-xs text-slate-100">
            {JSON.stringify(model.params, null, 2)}
          </pre>
        </Card>
      </div>
    </PageShell>
  );
}
