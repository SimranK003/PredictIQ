"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DriftFeatureResult } from "@/lib/types";

/** Score-to-threshold ratio per feature — a quick visual scan of which
 * features are furthest past their configured threshold. Bars are
 * colored by drift status, but the table below (the source of truth)
 * repeats the same status as text, never color-only. */
export function DriftFeatureChart({ features }: { features: DriftFeatureResult[] }) {
  const data = features
    .filter((f) => f.score !== null)
    .map((f) => ({
      feature: f.feature,
      ratio: f.threshold > 0 ? (f.score as number) / f.threshold : 0,
      driftDetected: f.drift_detected,
    }));

  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={Math.max(220, data.length * 28)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fontSize: 11, fill: "#64748b" }}
          axisLine={{ stroke: "#e2e8f0" }}
          label={{ value: "score ÷ threshold", position: "insideBottom", offset: -2, fontSize: 11, fill: "#94a3b8" }}
        />
        <YAxis
          type="category"
          dataKey="feature"
          width={110}
          tick={{ fontSize: 11, fill: "#334155" }}
          axisLine={{ stroke: "#e2e8f0" }}
        />
        <Tooltip
          formatter={(value, _name, item) => [
            `${Number(value).toFixed(2)}x threshold`,
            item.payload.driftDetected ? "Drift detected" : "No drift",
          ]}
          contentStyle={{ fontSize: 12, borderRadius: 8, borderColor: "#e2e8f0" }}
        />
        <Bar dataKey="ratio" radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.feature} fill={entry.driftDetected ? "#dc2626" : "#94a3b8"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
