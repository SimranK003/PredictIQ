"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/States";
import { Pagination } from "@/components/ui/Pagination";
import { formatDateTime, formatPercent, truncateId } from "@/lib/format";
import { PredictionDetailPanel } from "@/components/predictions/PredictionDetailPanel";

const LIMIT = 25;

export default function PredictionsPage() {
  return (
    <Suspense fallback={<LoadingState label="Loading predictions…" />}>
      <PredictionsPageContent />
    </Suspense>
  );
}

function PredictionsPageContent() {
  const searchParams = useSearchParams();
  const initialModelVersionId = searchParams.get("model_version_id") ?? "";

  const [offset, setOffset] = useState(0);
  const [modelVersionFilter, setModelVersionFilter] = useState(initialModelVersionId);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [selectedPredictionId, setSelectedPredictionId] = useState<string | null>(null);

  const modelsQuery = useQuery({ queryKey: queryKeys.models(), queryFn: () => api.listModels() });
  const predictionsQuery = useQuery({
    queryKey: queryKeys.predictions({
      limit: LIMIT,
      offset,
      modelVersionId: modelVersionFilter || undefined,
      startDate: startDate || undefined,
      endDate: endDate || undefined,
    }),
    queryFn: () =>
      api.listPredictions({
        limit: LIMIT,
        offset,
        modelVersionId: modelVersionFilter || undefined,
        startDate: startDate ? new Date(startDate).toISOString() : undefined,
        endDate: endDate ? new Date(endDate).toISOString() : undefined,
      }),
    refetchInterval: 15_000,
  });

  const modelLabel = (id: string) =>
    modelsQuery.data?.find((m) => m.id === id)?.version_label ?? truncateId(id);

  return (
    <PageShell
      title="Predictions"
      description="Real inference history from the predictions table — every row is an actual scored request."
    >
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="flex flex-col text-xs text-slate-500">
          Model version
          <select
            value={modelVersionFilter}
            onChange={(e) => {
              setModelVersionFilter(e.target.value);
              setOffset(0);
            }}
            className="mt-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
          >
            <option value="">All versions</option>
            {modelsQuery.data?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.version_label} ({m.algorithm})
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col text-xs text-slate-500">
          From
          <input
            type="date"
            value={startDate}
            onChange={(e) => {
              setStartDate(e.target.value);
              setOffset(0);
            }}
            className="mt-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
          />
        </label>
        <label className="flex flex-col text-xs text-slate-500">
          To
          <input
            type="date"
            value={endDate}
            onChange={(e) => {
              setEndDate(e.target.value);
              setOffset(0);
            }}
            className="mt-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
          />
        </label>
        {(modelVersionFilter || startDate || endDate) && (
          <button
            type="button"
            onClick={() => {
              setModelVersionFilter("");
              setStartDate("");
              setEndDate("");
              setOffset(0);
            }}
            className="rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-500 hover:bg-slate-50"
          >
            Clear filters
          </button>
        )}
      </div>

      {predictionsQuery.isPending && <LoadingState label="Loading predictions…" />}
      {predictionsQuery.error && (
        <ErrorState
          message={
            predictionsQuery.error instanceof Error
              ? predictionsQuery.error.message
              : "Failed to load predictions."
          }
          onRetry={() => predictionsQuery.refetch()}
        />
      )}
      {predictionsQuery.data && predictionsQuery.data.items.length === 0 && (
        <EmptyState
          title="No predictions match these filters"
          description="Try a different model version or clear the date range."
        />
      )}

      {predictionsQuery.data && predictionsQuery.data.items.length > 0 && (
        <div className="flex flex-col gap-3">
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th scope="col" className="px-4 py-3">Prediction ID</th>
                  <th scope="col" className="px-4 py-3">Model version</th>
                  <th scope="col" className="px-4 py-3">Prediction</th>
                  <th scope="col" className="px-4 py-3">Probability</th>
                  <th scope="col" className="px-4 py-3">Timestamp</th>
                  <th scope="col" className="px-4 py-3">Request ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {predictionsQuery.data.items.map((prediction) => (
                  <tr
                    key={prediction.id}
                    onClick={() => setSelectedPredictionId(prediction.id)}
                    className="cursor-pointer hover:bg-slate-50"
                  >
                    <td className="px-4 py-3 font-mono text-xs text-indigo-700">
                      {truncateId(prediction.id)}
                    </td>
                    <td className="px-4 py-3 text-slate-600">
                      {modelLabel(prediction.model_version_id)}
                    </td>
                    <td className="px-4 py-3 font-medium text-slate-900">{prediction.prediction}</td>
                    <td className="px-4 py-3 font-mono text-slate-700">
                      {formatPercent(prediction.probability, 2)}
                    </td>
                    <td className="px-4 py-3 text-slate-500">{formatDateTime(prediction.created_at)}</td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-500">
                      {truncateId(prediction.request_id)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pagination
            total={predictionsQuery.data.total}
            limit={LIMIT}
            offset={offset}
            onOffsetChange={setOffset}
          />
        </div>
      )}

      {selectedPredictionId && (
        <PredictionDetailPanel
          predictionId={selectedPredictionId}
          onClose={() => setSelectedPredictionId(null)}
        />
      )}
    </PageShell>
  );
}
