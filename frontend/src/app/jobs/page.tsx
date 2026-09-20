"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { PageShell } from "@/components/layout/PageShell";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/States";
import { JobStatusBadge } from "@/components/ui/StatusBadge";
import { Pagination } from "@/components/ui/Pagination";
import { formatDateTime, formatDuration, truncateId } from "@/lib/format";
import type { Job } from "@/lib/types";

const LIMIT = 20;

function jobDuration(job: Job): number | null {
  if (!job.started_at || !job.finished_at) return null;
  return (new Date(job.finished_at).getTime() - new Date(job.started_at).getTime()) / 1000;
}

export default function JobsPage() {
  return (
    <Suspense fallback={<LoadingState label="Loading jobs…" />}>
      <JobsPageContent />
    </Suspense>
  );
}

function JobsPageContent() {
  const searchParams = useSearchParams();
  const highlightId = searchParams.get("highlight");
  const [offset, setOffset] = useState(0);
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");

  const jobsQuery = useQuery({
    queryKey: queryKeys.jobs({ jobType: typeFilter || undefined, status: statusFilter || undefined, limit: LIMIT, offset }),
    queryFn: () =>
      api.listJobs({
        jobType: typeFilter || undefined,
        status: statusFilter || undefined,
        limit: LIMIT,
        offset,
      }),
    // Poll while any job on this page is still active — running jobs
    // finish in single-digit seconds for this dataset size, so 4s keeps
    // the page feeling live without hammering the API.
    refetchInterval: (query) => {
      const hasActive = query.state.data?.items.some(
        (job) => job.status === "queued" || job.status === "running",
      );
      return hasActive ? 4_000 : false;
    },
  });
  const datasetsQuery = useQuery({ queryKey: queryKeys.datasets, queryFn: api.listDatasets });

  const datasetName = (id: string | null) => {
    if (!id) return "—";
    return datasetsQuery.data?.find((d) => d.id === id)?.filename ?? truncateId(id);
  };

  return (
    <PageShell
      title="Training Jobs"
      description="Every dataset validation → training → MLflow → model-registration run, queued or completed."
    >
      <div className="mb-4 flex flex-wrap gap-3">
        <select
          value={typeFilter}
          onChange={(e) => {
            setTypeFilter(e.target.value);
            setOffset(0);
          }}
          aria-label="Filter by job type"
          className="rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
        >
          <option value="">All types</option>
          <option value="train">Training</option>
          <option value="batch_predict">Batch prediction</option>
        </select>
        <select
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setOffset(0);
          }}
          aria-label="Filter by status"
          className="rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
        >
          <option value="">All statuses</option>
          <option value="queued">Queued</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      {jobsQuery.isPending && <LoadingState label="Loading jobs…" />}
      {jobsQuery.error && (
        <ErrorState
          message={jobsQuery.error instanceof Error ? jobsQuery.error.message : "Failed to load jobs."}
          onRetry={() => jobsQuery.refetch()}
        />
      )}
      {jobsQuery.data && jobsQuery.data.items.length === 0 && (
        <EmptyState title="No jobs found" description="Submit a training job from the Train page." />
      )}

      {jobsQuery.data && jobsQuery.data.items.length > 0 && (
        <div className="flex flex-col gap-3">
          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="w-full min-w-[960px] text-left text-sm">
              <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th scope="col" className="px-4 py-3">Job ID</th>
                  <th scope="col" className="px-4 py-3">Type</th>
                  <th scope="col" className="px-4 py-3">Dataset</th>
                  <th scope="col" className="px-4 py-3">Status</th>
                  <th scope="col" className="px-4 py-3">Created</th>
                  <th scope="col" className="px-4 py-3">Started</th>
                  <th scope="col" className="px-4 py-3">Completed</th>
                  <th scope="col" className="px-4 py-3">Duration</th>
                  <th scope="col" className="px-4 py-3">Model versions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {jobsQuery.data.items.map((job) => (
                  <tr
                    key={job.id}
                    className={job.id === highlightId ? "bg-indigo-50" : "hover:bg-slate-50"}
                  >
                    <td className="px-4 py-3 font-mono text-xs text-slate-600">{truncateId(job.id)}</td>
                    <td className="px-4 py-3 capitalize text-slate-600">
                      {job.job_type.replace(/_/g, " ")}
                    </td>
                    <td className="px-4 py-3 text-slate-600">{datasetName(job.dataset_id)}</td>
                    <td className="px-4 py-3">
                      <JobStatusBadge status={job.status} />
                    </td>
                    <td className="px-4 py-3 text-slate-500">{formatDateTime(job.created_at)}</td>
                    <td className="px-4 py-3 text-slate-500">{formatDateTime(job.started_at)}</td>
                    <td className="px-4 py-3 text-slate-500">{formatDateTime(job.finished_at)}</td>
                    <td className="px-4 py-3 text-slate-500">{formatDuration(jobDuration(job))}</td>
                    <td className="px-4 py-3">
                      {job.model_version_ids && job.model_version_ids.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {job.model_version_ids.map((id) => (
                            <Link
                              key={id}
                              href={`/models/${id}`}
                              className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-indigo-700 hover:bg-indigo-100"
                            >
                              {truncateId(id, 6)}
                            </Link>
                          ))}
                        </div>
                      ) : job.model_version ? (
                        <span className="text-slate-600">{job.model_version}</span>
                      ) : (
                        <span className="text-slate-400">
                          {job.status === "failed" ? job.error_message?.slice(0, 40) ?? "—" : "—"}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pagination
            total={jobsQuery.data.total}
            limit={LIMIT}
            offset={offset}
            onOffsetChange={setOffset}
          />
        </div>
      )}
    </PageShell>
  );
}
