"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModelUsage } from "@/lib/types";
import { EmptyState } from "@/components/ui/States";

/** Which model versions are actually serving traffic — the "is a stale
 * model still receiving predictions after promotion?" view, straight
 * from GET /monitoring/summary's model_version_usage (real DB counts,
 * never synthesized). */
export function ModelUsageChart({ usage }: { usage: ModelUsage[] }) {
  if (usage.length === 0) {
    return <EmptyState title="No predictions recorded yet" />;
  }

  const data = usage.map((u) => ({
    name: u.version_label ?? u.model_version_id.slice(0, 8),
    predictions: u.prediction_count,
    stage: u.stage ?? "unknown",
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#64748b" }} axisLine={{ stroke: "#e2e8f0" }} />
        <YAxis tick={{ fontSize: 12, fill: "#64748b" }} axisLine={{ stroke: "#e2e8f0" }} allowDecimals={false} />
        <Tooltip
          formatter={(value, _name, item) => [
            `${Number(value).toLocaleString()} predictions`,
            `${item.payload.name} (${item.payload.stage})`,
          ]}
          contentStyle={{ fontSize: 12, borderRadius: 8, borderColor: "#e2e8f0" }}
        />
        <Bar dataKey="predictions" fill="#4f46e5" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
