"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { HealthBadge } from "@/components/ui/StatusBadge";

/** Polled lightly (60s) in the top bar of every page — health rarely
 * changes second-to-second, so this avoids aggressive polling while
 * still surfacing a real, current backend status everywhere. */
export function HealthIndicator() {
  const { data, isError } = useQuery({
    queryKey: queryKeys.health,
    queryFn: api.getHealth,
    refetchInterval: 60_000,
  });

  if (isError) {
    return <HealthBadge status="unhealthy" />;
  }
  if (!data) {
    return <span className="text-xs text-slate-400">Checking health…</span>;
  }
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <HealthBadge status={data.status} />
      <span className="hidden sm:inline">
        API {data.status} · DB {data.database} · Redis {data.redis}
      </span>
    </div>
  );
}
