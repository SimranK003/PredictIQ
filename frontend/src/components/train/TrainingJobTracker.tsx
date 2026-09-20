"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { JobStatusBadge } from "@/components/ui/StatusBadge";
import { ErrorState, LoadingState } from "@/components/ui/States";

const STEPS = ["queued", "running", "completed"] as const;

/** Tracks a just-submitted job in place — no page reload — polling
 * every 2s while active, matching the queued -> running -> completed/
 * failed lifecycle the job actually goes through. */
export function TrainingJobTracker({ jobId }: { jobId: string }) {
  const jobQuery = useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: () => api.getJob(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "running" ? 2_000 : false;
    },
  });

  if (jobQuery.isPending) {
    return <LoadingState label="Loading job status…" />;
  }

  if (jobQuery.error || !jobQuery.data) {
    return (
      <ErrorState
        message={jobQuery.error instanceof Error ? jobQuery.error.message : "Failed to load job status."}
        onRetry={() => jobQuery.refetch()}
      />
    );
  }

  const job = jobQuery.data;
  const stepIndex = job.status === "failed" ? -1 : STEPS.indexOf(job.status as (typeof STEPS)[number]);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="mb-4 flex items-center justify-between">
        <p className="font-mono text-xs text-slate-500">Job {job.id.slice(0, 8)}</p>
        <JobStatusBadge status={job.status} />
      </div>

      <ol className="flex items-center gap-2" aria-label="Job progress">
        {STEPS.map((step, index) => {
          const reached = job.status === "failed" ? index === 0 : index <= stepIndex;
          return (
            <li key={step} className="flex flex-1 items-center gap-2">
              <span
                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                  reached ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-400"
                }`}
                aria-hidden="true"
              >
                {index + 1}
              </span>
              <span className={`text-xs capitalize ${reached ? "text-slate-800" : "text-slate-400"}`}>
                {step}
              </span>
              {index < STEPS.length - 1 && <span className="h-px flex-1 bg-slate-200" aria-hidden="true" />}
            </li>
          );
        })}
      </ol>

      {job.status === "failed" && (
        <p className="mt-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
          {job.error_message ?? "Training failed with no recorded error message."}
        </p>
      )}

      {job.status === "completed" && job.result?.candidate_model_version_ids && (
        <div className="mt-4 rounded-md bg-emerald-50 px-3 py-3 text-sm text-emerald-800">
          <p className="font-medium">
            {job.result.candidate_model_version_ids.length} candidate model(s) registered.
          </p>
          {job.result.best_model_version_id && (
            <p className="mt-1">
              Best: {job.result.best_algorithm} (ROC-AUC{" "}
              {job.result.best_test_roc_auc?.toFixed(4)}) —{" "}
              <Link
                href={`/models/${job.result.best_model_version_id}`}
                className="font-medium underline"
              >
                view model
              </Link>
            </p>
          )}
        </div>
      )}

      <p className="mt-4 text-xs text-slate-400">
        <Link href={`/jobs?highlight=${job.id}`} className="underline">
          View in Training Jobs
        </Link>
      </p>
    </div>
  );
}
