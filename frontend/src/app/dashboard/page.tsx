"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { Card, MetricTile } from "@/components/ui/Card";
import { ErrorState, LoadingState } from "@/components/ui/States";
import { HealthBadge } from "@/components/ui/StatusBadge";
import { formatDateTime, formatRelativeTime } from "@/lib/format";
import { ModelUsageChart } from "@/components/dashboard/ModelUsageChart";

export default function DashboardPage() {
  const summaryQuery = useQuery({
    queryKey: queryKeys.monitoringSummary,
    queryFn: api.getMonitoringSummary,
    refetchInterval: 30_000,
  });
  const healthQuery = useQuery({
    queryKey: queryKeys.health,
    queryFn: api.getHealth,
    refetchInterval: 30_000,
  });

  const isLoading = summaryQuery.isPending || healthQuery.isPending;
  const error = summaryQuery.error ?? healthQuery.error;

  return (
    <PageShell
      title="Overview"
      description="The current state of the deployed churn-prediction system, from real API responses."
    >
      {isLoading && <LoadingState label="Loading system overview…" />}

      {error && !isLoading && (
        <ErrorState
          message={error instanceof Error ? error.message : "Failed to load the overview."}
          onRetry={() => {
            summaryQuery.refetch();
            healthQuery.refetch();
          }}
        />
      )}

      {summaryQuery.data && healthQuery.data && (
        <div className="flex flex-col gap-6">
          {/* Production model + system health */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card title="Production model">
              {summaryQuery.data.current_production_model ? (
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-semibold text-slate-900">
                      {summaryQuery.data.current_production_model.version_label}
                    </p>
                    <p className="text-sm text-slate-500">
                      {summaryQuery.data.current_production_model.algorithm}
                    </p>
                    <p className="mt-1 text-xs text-slate-400">
                      Promoted {formatRelativeTime(summaryQuery.data.current_production_model.promoted_at)}
                    </p>
                  </div>
                  <Link
                    href={`/models/${summaryQuery.data.current_production_model.model_version_id}`}
                    className="rounded-md border border-slate-200 px-3 py-1.5 text-sm font-medium text-indigo-700 hover:bg-indigo-50"
                  >
                    View details
                  </Link>
                </div>
              ) : (
                <p className="text-sm text-slate-500">
                  No production model available. Promote a candidate from{" "}
                  <Link href="/models" className="text-indigo-600 underline">
                    the model registry
                  </Link>
                  .
                </p>
              )}
            </Card>

            <Card title="System health">
              <dl className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <dt className="text-xs text-slate-500">API</dt>
                  <dd className="mt-1">
                    <HealthBadge status={healthQuery.data.status} />
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-slate-500">Database</dt>
                  <dd className="mt-1">
                    <HealthBadge status={healthQuery.data.database} />
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-slate-500">Redis</dt>
                  <dd className="mt-1">
                    <HealthBadge status={healthQuery.data.redis} />
                  </dd>
                </div>
              </dl>
            </Card>
          </div>

          {/* Key metrics */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricTile
              label={`Predictions (${summaryQuery.data.recent_window_hours}h)`}
              value={summaryQuery.data.recent_prediction_count.toLocaleString()}
            />
            <MetricTile
              label="Prediction errors (this process)"
              value={summaryQuery.data.recent_prediction_error_count.toLocaleString()}
              hint="Prometheus counter, scoped to this API process"
            />
            <MetricTile
              label="Active training jobs"
              value={summaryQuery.data.active_training_jobs}
            />
            <MetricTile
              label="Drift status"
              value={<span className="capitalize">{summaryQuery.data.drift_status.replace(/_/g, " ")}</span>}
              hint={
                <Link href="/monitoring" className="text-indigo-600 underline">
                  View details
                </Link>
              }
            />
          </div>

          {/* Model version usage */}
          <Card
            title="Prediction volume by model version"
            action={
              <Link href="/monitoring" className="text-sm text-indigo-600 hover:underline">
                Full monitoring →
              </Link>
            }
          >
            <p className="mb-3 text-xs text-slate-400">
              Real counts from the predictions table, grouped by exact model version — this is how a
              stale model still serving traffic after a promotion would show up.
            </p>
            <ModelUsageChart usage={summaryQuery.data.model_version_usage} />
          </Card>

          {/* Recent failed jobs */}
          <Card title="Recent failed jobs (last 24h)">
            {summaryQuery.data.recent_failed_jobs.length === 0 ? (
              <p className="text-sm text-slate-500">No failed jobs in the last 24 hours.</p>
            ) : (
              <ul className="divide-y divide-slate-100">
                {summaryQuery.data.recent_failed_jobs.map((job) => (
                  <li key={job.job_id} className="py-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs text-slate-500">{job.job_id.slice(0, 8)}</span>
                      <span className="text-xs text-slate-400">{formatDateTime(job.created_at)}</span>
                    </div>
                    <p className="mt-0.5 text-slate-700">
                      <span className="font-medium capitalize">{job.job_type.replace(/_/g, " ")}</span>
                      {": "}
                      {job.error_message ?? "No error message recorded."}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          {/* Training job stats summary */}
          <Card title="Training job statistics (all time)">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <MetricTile label="Total jobs" value={summaryQuery.data.training_job_stats.total_jobs} />
              <MetricTile
                label="Completed"
                value={summaryQuery.data.training_job_stats.completed_jobs}
              />
              <MetricTile label="Failed" value={summaryQuery.data.training_job_stats.failed_jobs} />
              <MetricTile
                label="Avg. duration"
                value={
                  summaryQuery.data.training_job_stats.average_training_duration_seconds !== null
                    ? `${summaryQuery.data.training_job_stats.average_training_duration_seconds.toFixed(1)}s`
                    : "—"
                }
              />
            </div>
          </Card>

          <p className="text-xs text-slate-400">
            &ldquo;Predictions this process&rdquo; figures elsewhere in this dashboard reflect only
            this API server process&apos;s in-memory Prometheus counters — a separate Celery worker
            process (e.g. for large async batches) increments its own, invisible here. See{" "}
            <Link href="/monitoring" className="underline">
              Monitoring
            </Link>{" "}
            for the real, database-backed prediction volume.
          </p>
        </div>
      )}
    </PageShell>
  );
}
