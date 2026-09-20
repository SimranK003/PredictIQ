"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { LoadingState, ErrorState } from "@/components/ui/States";
import { formatDateTime, formatPercent } from "@/lib/format";

/**
 * The one place raw input_features is shown — a deliberate per-record
 * drill-down the user explicitly requested by clicking a row, not
 * something dumped into the list table (per the "don't expose raw
 * input_features unnecessarily" requirement).
 */
export function PredictionDetailPanel({
  predictionId,
  onClose,
}: {
  predictionId: string;
  onClose: () => void;
}) {
  const detailQuery = useQuery({
    queryKey: queryKeys.prediction(predictionId),
    queryFn: () => api.getPrediction(predictionId),
  });

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="prediction-detail-title"
      className="fixed inset-0 z-50 flex justify-end bg-slate-900/40"
    >
      <div className="h-full w-full max-w-md overflow-y-auto bg-white p-6 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 id="prediction-detail-title" className="text-base font-semibold text-slate-900">
            Prediction detail
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            ✕
          </button>
        </div>

        {detailQuery.isPending && <LoadingState label="Loading prediction…" />}
        {detailQuery.error && (
          <ErrorState
            message={detailQuery.error instanceof Error ? detailQuery.error.message : "Failed to load."}
            onRetry={() => detailQuery.refetch()}
          />
        )}

        {detailQuery.data && (
          <div className="flex flex-col gap-4 text-sm">
            <dl className="grid grid-cols-2 gap-3">
              <div>
                <dt className="text-xs text-slate-500">Prediction</dt>
                <dd className="mt-1 font-semibold text-slate-900">{detailQuery.data.prediction}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Probability</dt>
                <dd className="mt-1 font-mono text-slate-900">
                  {formatPercent(detailQuery.data.probability, 2)}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Model version</dt>
                <dd className="mt-1 text-slate-900">{detailQuery.data.model_version_label}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Algorithm</dt>
                <dd className="mt-1 text-slate-900">{detailQuery.data.algorithm}</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Latency</dt>
                <dd className="mt-1 font-mono text-slate-900">
                  {detailQuery.data.latency_ms.toFixed(2)}ms
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Timestamp</dt>
                <dd className="mt-1 text-slate-900">{formatDateTime(detailQuery.data.created_at)}</dd>
              </div>
              <div className="col-span-2">
                <dt className="text-xs text-slate-500">Request ID</dt>
                <dd className="mt-1 break-all font-mono text-xs text-slate-700">
                  {detailQuery.data.request_id}
                </dd>
              </div>
            </dl>

            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
                Input features
              </p>
              <pre className="max-h-80 overflow-auto rounded-md bg-slate-900 p-3 text-xs text-slate-100">
                {JSON.stringify(detailQuery.data.input_features, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
