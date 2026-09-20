import Link from "next/link";
import { truncateId } from "@/lib/format";

interface LineageStep {
  label: string;
  value: string;
  href?: string;
  mono?: boolean;
}

/** Renders the Dataset -> Training Job -> MLflow Run -> Model Version ->
 * Predictions chain as a simple vertical flow — the concrete answer to
 * "which data and which run produced this model," not just an abstract
 * diagram. */
export function LineageDiagram({ steps }: { steps: LineageStep[] }) {
  return (
    <ol className="flex flex-col">
      {steps.map((step, index) => (
        <li key={step.label} className="relative pl-8">
          {index < steps.length - 1 && (
            <span
              aria-hidden="true"
              className="absolute left-[11px] top-6 h-full w-px bg-slate-200"
            />
          )}
          <span
            aria-hidden="true"
            className="absolute left-0 top-1 flex h-6 w-6 items-center justify-center rounded-full border-2 border-indigo-500 bg-white text-xs font-semibold text-indigo-600"
          >
            {index + 1}
          </span>
          <div className="pb-6">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
              {step.label}
            </p>
            {step.href ? (
              <Link
                href={step.href}
                className={`text-sm text-indigo-600 hover:underline ${step.mono ? "font-mono" : ""}`}
              >
                {step.value}
              </Link>
            ) : (
              <p className={`text-sm text-slate-800 ${step.mono ? "font-mono" : ""}`}>{step.value}</p>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

export function buildLineageSteps(opts: {
  datasetFilename: string;
  datasetId: string;
  trainingJobId: string | null;
  mlflowRunId: string;
  modelVersionLabel: string;
  modelVersionId: string;
  predictionCount: number;
}): LineageStep[] {
  return [
    { label: "Dataset", value: opts.datasetFilename, href: `/models?dataset=${opts.datasetId}` },
    {
      label: "Training job",
      value: opts.trainingJobId ? truncateId(opts.trainingJobId, 12) : "None (CLI-trained, Phase 2)",
      href: opts.trainingJobId ? `/jobs?highlight=${opts.trainingJobId}` : undefined,
      mono: Boolean(opts.trainingJobId),
    },
    { label: "MLflow run", value: truncateId(opts.mlflowRunId, 16), mono: true },
    { label: "Model version", value: opts.modelVersionLabel, href: `/models/${opts.modelVersionId}` },
    {
      label: "Predictions",
      value: `${opts.predictionCount.toLocaleString()} recorded`,
      href: `/predictions?model_version_id=${opts.modelVersionId}`,
    },
  ];
}
