import clsx from "clsx";

type Tone = "success" | "warning" | "danger" | "neutral" | "info";

const TONE_STYLES: Record<Tone, { bg: string; text: string; dot: string }> = {
  success: { bg: "bg-emerald-50", text: "text-emerald-700", dot: "bg-emerald-500" },
  warning: { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  danger: { bg: "bg-red-50", text: "text-red-700", dot: "bg-red-500" },
  neutral: { bg: "bg-slate-100", text: "text-slate-600", dot: "bg-slate-400" },
  info: { bg: "bg-indigo-50", text: "text-indigo-700", dot: "bg-indigo-500" },
};

/**
 * A status indicator that never relies on color alone: every badge pairs
 * a shape (the dot) and a text label with its color, so the meaning is
 * legible in grayscale or to someone who can't distinguish red/green.
 */
export function StatusBadge({ tone, label }: { tone: Tone; label: string }) {
  const styles = TONE_STYLES[tone];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
        styles.bg,
        styles.text,
      )}
    >
      <span className={clsx("status-dot", styles.dot)} aria-hidden="true" />
      {label}
    </span>
  );
}

const STAGE_TONE: Record<string, Tone> = {
  production: "success",
  candidate: "info",
  previous: "warning",
  archived: "neutral",
};

export function StageBadge({ stage }: { stage: string }) {
  return <StatusBadge tone={STAGE_TONE[stage] ?? "neutral"} label={stage} />;
}

const JOB_STATUS_TONE: Record<string, Tone> = {
  queued: "neutral",
  running: "info",
  completed: "success",
  failed: "danger",
};

export function JobStatusBadge({ status }: { status: string }) {
  return <StatusBadge tone={JOB_STATUS_TONE[status] ?? "neutral"} label={status} />;
}

const HEALTH_TONE: Record<string, Tone> = {
  ok: "success",
  degraded: "warning",
  unhealthy: "danger",
  available: "success",
  unavailable: "danger",
};

export function HealthBadge({ status }: { status: string }) {
  return <StatusBadge tone={HEALTH_TONE[status] ?? "neutral"} label={status} />;
}
