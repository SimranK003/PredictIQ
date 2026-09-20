import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";

describe("api client error handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("throws an ApiError with the backend's detail message on an HTTP error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Dataset not found." }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(api.getDataset("missing-id")).rejects.toMatchObject({
      message: "Dataset not found.",
      status: 404,
    });
  });

  it("throws an ApiError when the response body is not valid JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("<html>502 Bad Gateway</html>", {
          status: 200,
          headers: { "Content-Type": "text/html" },
        }),
      ),
    );

    await expect(api.getDataset("some-id")).rejects.toBeInstanceOf(ApiError);
  });

  it("throws a clear ApiError when the network request itself fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    await expect(api.getDataset("some-id")).rejects.toMatchObject({
      status: 0,
    });
  });

  it("throws a timeout-specific ApiError when the request is aborted", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted.", "AbortError"));
          });
        });
      }),
    );

    const pending = expect(api.getDataset("some-id")).rejects.toBeInstanceOf(ApiError);
    await vi.advanceTimersByTimeAsync(10_000);
    await pending;
    vi.useRealTimers();
  });

  it("returns parsed JSON on a successful response", async () => {
    const dataset = { id: "d1", filename: "churn.csv", is_valid: true };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(dataset), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(api.getDataset("d1")).resolves.toMatchObject({ id: "d1" });
  });
});
