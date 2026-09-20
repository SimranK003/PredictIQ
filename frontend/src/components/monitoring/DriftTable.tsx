import { StatusBadge } from "@/components/ui/StatusBadge";
import type { DriftFeatureResult } from "@/lib/types";

export function DriftTable({ features }: { features: DriftFeatureResult[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th scope="col" className="px-4 py-2.5">Feature</th>
            <th scope="col" className="px-4 py-2.5">Test</th>
            <th scope="col" className="px-4 py-2.5">Score</th>
            <th scope="col" className="px-4 py-2.5">Threshold</th>
            <th scope="col" className="px-4 py-2.5">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {features.map((feature) => (
            <tr key={feature.feature}>
              <td className="px-4 py-2.5 font-medium text-slate-800">{feature.feature}</td>
              <td className="px-4 py-2.5 text-slate-500">{feature.test}</td>
              <td className="px-4 py-2.5 font-mono text-slate-700">
                {feature.score === null ? "—" : feature.score.toFixed(4)}
              </td>
              <td className="px-4 py-2.5 font-mono text-slate-500">{feature.threshold.toFixed(2)}</td>
              <td className="px-4 py-2.5">
                {feature.score === null ? (
                  <StatusBadge tone="neutral" label="No data" />
                ) : feature.drift_detected ? (
                  <StatusBadge
                    tone={feature.severity === "critical" ? "danger" : "warning"}
                    label={`Drift detected${feature.severity ? ` (${feature.severity})` : ""}`}
                  />
                ) : (
                  <StatusBadge tone="success" label="No drift" />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
