"use client";

import Link from "next/link";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/States";
import { StageBadge } from "@/components/ui/StatusBadge";
import { formatDateTime, formatPercent, truncateId } from "@/lib/format";
import type { ModelStage } from "@/lib/types";

const STAGE_FILTERS: { value: ModelStage | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "production", label: "Production" },
  { value: "candidate", label: "Candidate" },
  { value: "previous", label: "Previous" },
  { value: "archived", label: "Archived" },
];

export default function ModelsPage() {
  const [stageFilter, setStageFilter] = useState<ModelStage | "all">("all");

  const modelsQuery = useQuery({
    queryKey: queryKeys.models(stageFilter === "all" ? undefined : stageFilter),
    queryFn: () => api.listModels(stageFilter === "all" ? undefined : stageFilter),
  });
  const datasetsQuery = useQuery({
    queryKey: queryKeys.datasets,
    queryFn: api.listDatasets,
  });

  const datasetName = (datasetId: string) =>
    datasetsQuery.data?.find((d) => d.id === datasetId)?.filename ?? truncateId(datasetId);

  return (
    <PageShell
      title="Model Registry"
      description="Every trained model version — candidate, production, previous, or archived — with the metrics that decide promotion."
    >
      <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filter by lifecycle stage">
        {STAGE_FILTERS.map((filter) => (
          <button
            key={filter.value}
            type="button"
            aria-pressed={stageFilter === filter.value}
            onClick={() => setStageFilter(filter.value)}
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              stageFilter === filter.value
                ? "bg-indigo-600 text-white"
                : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
            }`}
          >
            {filter.label}
          </button>
        ))}
      </div>

      {modelsQuery.isPending && <LoadingState label="Loading model versions…" />}
      {modelsQuery.error && (
        <ErrorState
          message={modelsQuery.error instanceof Error ? modelsQuery.error.message : "Failed to load models."}
          onRetry={() => modelsQuery.refetch()}
        />
      )}
      {modelsQuery.data && modelsQuery.data.length === 0 && (
        <EmptyState
          title="No model versions match this filter"
          description="Train a model from the Train page to create the first candidate."
        />
      )}

      {modelsQuery.data && modelsQuery.data.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
          <table className="w-full min-w-[900px] text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th scope="col" className="px-4 py-3">Version</th>
                <th scope="col" className="px-4 py-3">Algorithm</th>
                <th scope="col" className="px-4 py-3">Stage</th>
                <th scope="col" className="px-4 py-3">ROC-AUC</th>
                <th scope="col" className="px-4 py-3">Precision</th>
                <th scope="col" className="px-4 py-3">Recall</th>
                <th scope="col" className="px-4 py-3">F1</th>
                <th scope="col" className="px-4 py-3">Dataset</th>
                <th scope="col" className="px-4 py-3">Created</th>
                <th scope="col" className="px-4 py-3">Training job</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {modelsQuery.data.map((model) => (
                <tr key={model.id} className="hover:bg-slate-50">
                  <td className="px-4 py-3 font-medium text-slate-900">
                    <Link href={`/models/${model.id}`} className="text-indigo-600 hover:underline">
                      {model.version_label}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-slate-600">{model.algorithm}</td>
                  <td className="px-4 py-3">
                    <StageBadge stage={model.stage} />
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-700">
                    {formatPercent(model.metrics.roc_auc)}
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-700">
                    {formatPercent(model.metrics.precision)}
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-700">
                    {formatPercent(model.metrics.recall)}
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-700">
                    {formatPercent(model.metrics.f1)}
                  </td>
                  <td className="px-4 py-3 text-slate-600" title={model.dataset_id}>
                    {datasetName(model.dataset_id)}
                  </td>
                  <td className="px-4 py-3 text-slate-500">{formatDateTime(model.created_at)}</td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-500">
                    {model.training_job_id ? (
                      <Link href={`/jobs?highlight=${model.training_job_id}`} className="hover:underline">
                        {truncateId(model.training_job_id)}
                      </Link>
                    ) : (
                      <span title="Registered before async training jobs existed (Phase 2 CLI path)">
                        —
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PageShell>
  );
}
