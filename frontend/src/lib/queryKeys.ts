/** Central React Query key factory — avoids typo'd/inconsistent cache
 * keys scattered across components, and gives mutations one obvious
 * place to invalidate from.
 */
export const queryKeys = {
  health: ["health"] as const,
  models: (stage?: string) => ["models", stage ?? "all"] as const,
  model: (id: string) => ["models", id] as const,
  productionModel: ["models", "production"] as const,
  datasets: ["datasets"] as const,
  dataset: (id: string) => ["datasets", id] as const,
  jobs: (filters?: { jobType?: string; status?: string; limit?: number; offset?: number }) =>
    ["jobs", filters ?? {}] as const,
  job: (id: string) => ["jobs", id] as const,
  predictions: (filters?: {
    limit?: number;
    offset?: number;
    modelVersionId?: string;
    startDate?: string;
    endDate?: string;
  }) => ["predictions", filters ?? {}] as const,
  prediction: (id: string) => ["predictions", id] as const,
  monitoringSummary: ["monitoring", "summary"] as const,
  modelUsage: ["monitoring", "model-usage"] as const,
  trainingJobStats: ["monitoring", "jobs"] as const,
  drift: (opts?: { modelVersionId?: string; window?: string; feature?: string }) =>
    ["monitoring", "drift", opts ?? {}] as const,
};
