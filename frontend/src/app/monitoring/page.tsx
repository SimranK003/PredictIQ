"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { Card, MetricTile } from "@/components/ui/Card";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/States";
import { formatDateTime, formatRelativeTime } from "@/lib/format";
import { ModelUsageChart } from "@/components/dashboard/ModelUsageChart";
import { DriftTable } from "@/components/monitoring/DriftTable";
import { DriftFeatureChart } from "@/components/monitoring/DriftFeatureChart";

const WINDOWS = [
  { value: "", label: "All time" },
  { value: "24h", label: "Last 24 hours" },
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
];

export default function MonitoringPage() {
  const [modelVersionId, setModelVersionId] = useState<string>("");
  const [window, setWindow] = useState<string>("");

  const modelsQuery = useQuery({ queryKey: queryKeys.models(), queryFn: () => api.listModels() });
  const summaryQuery = useQuery({
    queryKey: queryKeys.monitoringSummary,
    queryFn: api.getMonitoringSummary,
    refetchInterval: 30_000,
  });
  const driftQuery = useQuery({
    queryKey: queryKeys.drift({ modelVersionId: modelVersionId || undefined, window: window || undefined }),
    queryFn: () => api.getDrift({ modelVersionId: modelVersionId || undefined, window: window || undefined }),
    refetchInterval: 30_000,
  });

  return (
    <PageShell
      title="Monitoring"
      description="Prediction volume and real statistical drift detection — see docs/monitoring.md for methodology."
    >
      <div className="flex flex-col gap-6">
        {/* Prediction monitoring */}
        <Card title="Prediction volume">
          {summaryQuery.isPending && <LoadingState label="Loading prediction volume…" />}
          {summaryQuery.error && (
            <ErrorState
              message={
                summaryQuery.error instanceof Error ? summaryQuery.error.message : "Failed to load."
              }
              onRetry={() => summaryQuery.refetch()}
            />
          )}
          {summaryQuery.data && (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <MetricTile
                  label={`Predictions (last ${summaryQuery.data.recent_window_hours}h)`}
                  value={summaryQuery.data.recent_prediction_count.toLocaleString()}
                />
                <MetricTile
                  label="Total predictions (all versions)"
                  value={summaryQuery.data.model_version_usage
                    .reduce((sum, u) => sum + u.prediction_count, 0)
                    .toLocaleString()}
                  hint="Real count from the predictions table"
                />
                <MetricTile
                  label="Prediction errors (this process)"
                  value={summaryQuery.data.recent_prediction_error_count}
                  hint="Prometheus counter — this API process only"
                />
              </div>
              <ModelUsageChart usage={summaryQuery.data.model_version_usage} />
              {summaryQuery.data.model_version_usage.length > 0 && (
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full min-w-[640px] text-left text-sm">
                    <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                      <tr>
                        <th scope="col" className="px-4 py-2.5">Version</th>
                        <th scope="col" className="px-4 py-2.5">Stage</th>
                        <th scope="col" className="px-4 py-2.5">Predictions</th>
                        <th scope="col" className="px-4 py-2.5">% of total</th>
                        <th scope="col" className="px-4 py-2.5">First seen</th>
                        <th scope="col" className="px-4 py-2.5">Last seen</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {summaryQuery.data.model_version_usage.map((usage) => (
                        <tr key={usage.model_version_id}>
                          <td className="px-4 py-2.5 font-medium text-slate-800">
                            {usage.version_label}
                          </td>
                          <td className="px-4 py-2.5 capitalize text-slate-500">{usage.stage}</td>
                          <td className="px-4 py-2.5 font-mono text-slate-700">
                            {usage.prediction_count.toLocaleString()}
                          </td>
                          <td className="px-4 py-2.5 font-mono text-slate-700">
                            {usage.percentage_of_total}%
                          </td>
                          <td className="px-4 py-2.5 text-slate-500">
                            {formatDateTime(usage.first_prediction_at)}
                          </td>
                          <td className="px-4 py-2.5 text-slate-500" title={usage.last_prediction_at}>
                            {formatRelativeTime(usage.last_prediction_at)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Drift monitoring */}
        <Card title="Data drift">
          <p className="mb-4 rounded-md bg-indigo-50 px-3 py-2 text-xs text-indigo-800">
            Data drift indicates a change in input distribution; it does not directly measure model
            accuracy.
          </p>

          <div className="mb-4 flex flex-wrap gap-3">
            <label className="flex flex-col text-xs text-slate-500">
              Model version
              <select
                value={modelVersionId}
                onChange={(e) => setModelVersionId(e.target.value)}
                className="mt-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
              >
                <option value="">Current production</option>
                {modelsQuery.data?.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.version_label} ({m.algorithm})
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col text-xs text-slate-500">
              Time window
              <select
                value={window}
                onChange={(e) => setWindow(e.target.value)}
                className="mt-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
              >
                {WINDOWS.map((w) => (
                  <option key={w.value} value={w.value}>
                    {w.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {driftQuery.isPending && <LoadingState label="Computing drift…" />}
          {driftQuery.error && (
            <ErrorState
              message={driftQuery.error instanceof Error ? driftQuery.error.message : "Failed to load drift."}
              onRetry={() => driftQuery.refetch()}
            />
          )}

          {driftQuery.data?.status === "no_production_model" && (
            <EmptyState
              title="No production model available"
              description="Promote a candidate from the Model Registry to enable drift monitoring."
            />
          )}

          {driftQuery.data?.status === "insufficient_data" && (
            <EmptyState
              title="Insufficient data for drift analysis"
              description={`${driftQuery.data.sample_size} of ${driftQuery.data.minimum_required} required predictions collected for ${driftQuery.data.model_version}. This is an honest "not enough data yet" — not a fabricated score.`}
            />
          )}

          {driftQuery.data?.status === "ok" && (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <MetricTile label="Model version" value={driftQuery.data.model_version} />
                <MetricTile label="Sample size" value={driftQuery.data.sample_size.toLocaleString()} />
                <MetricTile
                  label="Reference dataset rows"
                  value={driftQuery.data.reference_n_rows.toLocaleString()}
                  hint={`hash ${driftQuery.data.reference_dataset_content_hash.slice(0, 10)}…`}
                />
                <MetricTile
                  label="Overall status"
                  value={driftQuery.data.drift_detected ? "Drift detected" : "No drift"}
                />
              </div>
              <DriftFeatureChart features={driftQuery.data.features} />
              <DriftTable features={driftQuery.data.features} />
            </div>
          )}
        </Card>
      </div>
    </PageShell>
  );
}
