/**
 * Single typed API client for the PredictIQ backend. Every request the
 * dashboard makes goes through here — no scattered fetch() calls in
 * components — so error handling, timeouts, and response typing are
 * consistent everywhere.
 */

import type {
  AuthUser,
  DriftReport,
  HealthStatus,
  Job,
  ModelUsage,
  ModelVersionDetail,
  ModelVersionSummary,
  MonitoringSummary,
  PaginatedJobs,
  PaginatedPredictions,
  PredictionDetail,
  DatasetDetail,
  DatasetSummary,
  TrainingJobStats,
  TrainRequest,
  TrainResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8123";
const DEFAULT_TIMEOUT_MS = 10_000;

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function extractDetailMessage(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (
    detail &&
    typeof detail === "object" &&
    "message" in detail &&
    typeof (detail as { message: unknown }).message === "string"
  ) {
    return (detail as { message: string }).message;
  }
  return null;
}

async function request<T>(
  path: string,
  init?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = init?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
      // The session cookie (HttpOnly, set by POST /auth/login) is on a
      // different origin from the frontend whenever ports differ, so it
      // won't be sent/stored without this — the backend's CORS config
      // (allow_credentials=True with an explicit origin allowlist, never
      // "*") is the other half of making that work.
      credentials: "include",
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(`Request to ${path} timed out after ${timeoutMs}ms.`, 0, null);
    }
    throw new ApiError(
      `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
      0,
      null,
    );
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    let detail: unknown = null;
    try {
      const body = await response.json();
      detail = body?.detail ?? body;
    } catch {
      // Response wasn't JSON (e.g. a proxy error page) — fall through with no detail.
    }
    const message = extractDetailMessage(detail) ?? `Request to ${path} failed (${response.status}).`;
    throw new ApiError(message, response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(`Received a malformed (non-JSON) response from ${path}.`, response.status, null);
  }
}

function buildQuery(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

export const api = {
  // --- Health ---
  getHealth: () => request<HealthStatus>("/health"),

  // --- Auth ---
  login: (email: string, password: string) =>
    request<AuthUser>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  getMe: () => request<AuthUser>("/auth/me"),

  // --- Models ---
  listModels: (stage?: string) =>
    request<ModelVersionSummary[]>(`/models${buildQuery({ stage })}`),
  getModel: (id: string) => request<ModelVersionDetail>(`/models/${id}`),
  getProductionModel: () => request<ModelVersionDetail>("/models/production"),
  promoteModel: (candidateId: string) =>
    request<ModelVersionDetail>("/models/promote", {
      method: "POST",
      body: JSON.stringify({ candidate_id: candidateId }),
    }),
  rollbackModel: () =>
    request<ModelVersionDetail>("/models/rollback", { method: "POST" }),

  // --- Datasets ---
  listDatasets: () => request<DatasetSummary[]>("/datasets"),
  getDataset: (id: string) => request<DatasetDetail>(`/datasets/${id}`),

  // --- Jobs ---
  listJobs: (opts?: { jobType?: string; status?: string; limit?: number; offset?: number }) =>
    request<PaginatedJobs>(
      `/jobs${buildQuery({
        job_type: opts?.jobType,
        status: opts?.status,
        limit: opts?.limit,
        offset: opts?.offset,
      })}`,
    ),
  getJob: (id: string) => request<Job>(`/jobs/${id}`),

  // --- Training ---
  submitTraining: (payload: TrainRequest) =>
    request<TrainResponse>("/train", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // --- Predictions ---
  listPredictions: (opts?: {
    limit?: number;
    offset?: number;
    modelVersionId?: string;
    startDate?: string;
    endDate?: string;
  }) =>
    request<PaginatedPredictions>(
      `/predictions${buildQuery({
        limit: opts?.limit,
        offset: opts?.offset,
        model_version_id: opts?.modelVersionId,
        start_date: opts?.startDate,
        end_date: opts?.endDate,
      })}`,
    ),
  getPrediction: (id: string) => request<PredictionDetail>(`/predictions/${id}`),

  // --- Monitoring ---
  getMonitoringSummary: () => request<MonitoringSummary>("/monitoring/summary"),
  getModelUsage: () => request<ModelUsage[]>("/monitoring/model-usage"),
  getTrainingJobStats: () => request<TrainingJobStats>("/monitoring/jobs"),
  getDrift: (opts?: { modelVersionId?: string; window?: string; feature?: string }) =>
    request<DriftReport>(
      `/monitoring/drift${buildQuery({
        model_version_id: opts?.modelVersionId,
        window: opts?.window,
        feature: opts?.feature,
      })}`,
    ),
};
