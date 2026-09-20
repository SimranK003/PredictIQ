import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import { AuthContext } from "@/components/auth/AuthProvider";
import { createTestQueryClient } from "@/test/queryClientWrapper";
import LoginPage from "./page";
import { api, ApiError } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: { login: vi.fn(), logout: vi.fn(), getMe: vi.fn() },
  ApiError: class MockApiError extends Error {
    status: number;
    constructor(message: string, status = 401) {
      super(message);
      this.status = status;
    }
  },
}));

function renderLoginPage(refetchUser = vi.fn()) {
  return render(
    <QueryClientProvider client={createTestQueryClient()}>
      <AuthContext.Provider
        value={{ user: null, isPending: false, refetchUser, logout: vi.fn() }}
      >
        <LoginPage />
      </AuthContext.Provider>
    </QueryClientProvider>,
  );
}

describe("LoginPage", () => {
  beforeEach(() => {
    vi.mocked(api.login).mockReset();
  });

  it("renders email and password fields and a submit button", () => {
    renderLoginPage();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("submits the entered credentials and refetches the session on success", async () => {
    const user = userEvent.setup();
    const refetchUser = vi.fn().mockResolvedValue(undefined);
    vi.mocked(api.login).mockResolvedValue({
      id: "u1",
      email: "admin@test.local",
      is_admin: true,
      created_at: "2026-01-01T00:00:00Z",
      last_login_at: null,
    });

    renderLoginPage(refetchUser);

    await user.type(screen.getByLabelText(/email/i), "admin@test.local");
    await user.type(screen.getByLabelText(/password/i), "correct-password");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(api.login).toHaveBeenCalledWith("admin@test.local", "correct-password");
    });
    await waitFor(() => {
      expect(refetchUser).toHaveBeenCalled();
    });
  });

  it("shows the backend's error message on a failed login and does not call refetchUser", async () => {
    const user = userEvent.setup();
    const refetchUser = vi.fn();
    vi.mocked(api.login).mockRejectedValue(new ApiError("Incorrect email or password.", 401, null));

    renderLoginPage(refetchUser);

    await user.type(screen.getByLabelText(/email/i), "admin@test.local");
    await user.type(screen.getByLabelText(/password/i), "wrong-password");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect email or password.");
    expect(refetchUser).not.toHaveBeenCalled();
  });

  it("disables the submit button while a login request is in flight", async () => {
    const user = userEvent.setup();
    let resolveLogin: (value: Awaited<ReturnType<typeof api.login>>) => void = () => {};
    vi.mocked(api.login).mockReturnValue(
      new Promise((resolve) => {
        resolveLogin = resolve;
      }),
    );

    renderLoginPage();
    await user.type(screen.getByLabelText(/email/i), "admin@test.local");
    await user.type(screen.getByLabelText(/password/i), "correct-password");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("button", { name: /signing in/i })).toBeDisabled();

    resolveLogin({
      id: "u1",
      email: "admin@test.local",
      is_admin: true,
      created_at: "2026-01-01T00:00:00Z",
      last_login_at: null,
    });
  });
});
