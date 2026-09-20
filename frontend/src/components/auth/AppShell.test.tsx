import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { AuthContext } from "./AuthProvider";
import { AppShell } from "./AppShell";
import type { AuthUser } from "@/lib/types";

const mockReplace = vi.fn();
let mockPathname = "/dashboard";

vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
  useRouter: () => ({ replace: mockReplace }),
}));

const ADMIN_USER: AuthUser = {
  id: "u1",
  email: "admin@test.local",
  is_admin: true,
  created_at: "2026-01-01T00:00:00Z",
  last_login_at: null,
};

function renderShell({
  user,
  isPending = false,
  pathname = "/dashboard",
}: {
  user: AuthUser | null;
  isPending?: boolean;
  pathname?: string;
}) {
  mockPathname = pathname;
  return render(
    <AuthContext.Provider
      value={{ user, isPending, refetchUser: vi.fn(), logout: vi.fn() }}
    >
      <AppShell>
        <div>Protected page content</div>
      </AppShell>
    </AuthContext.Provider>,
  );
}

describe("AppShell", () => {
  beforeEach(() => {
    mockReplace.mockReset();
  });

  it("shows a loading state, not page content, while auth status is pending", () => {
    renderShell({ user: null, isPending: true, pathname: "/dashboard" });
    expect(screen.getByRole("status", { name: /loading/i })).toBeInTheDocument();
    expect(screen.queryByText("Protected page content")).not.toBeInTheDocument();
  });

  it("redirects to /login and does not render content when there is no session", () => {
    renderShell({ user: null, isPending: false, pathname: "/dashboard" });
    expect(mockReplace).toHaveBeenCalledWith("/login");
    expect(screen.queryByText("Protected page content")).not.toBeInTheDocument();
  });

  it("redirects an already-authenticated user away from /login", () => {
    renderShell({ user: ADMIN_USER, isPending: false, pathname: "/login" });
    expect(mockReplace).toHaveBeenCalledWith("/dashboard");
  });

  it("renders the login page content directly (no sidebar) when unauthenticated on /login", () => {
    renderShell({ user: null, isPending: false, pathname: "/login" });
    expect(mockReplace).not.toHaveBeenCalled();
    expect(screen.getByText("Protected page content")).toBeInTheDocument();
    expect(screen.queryByText("PredictIQ")).not.toBeInTheDocument();
  });

  it("renders the sidebar and page content for an authenticated user on a protected route", () => {
    renderShell({ user: ADMIN_USER, isPending: false, pathname: "/dashboard" });
    expect(mockReplace).not.toHaveBeenCalled();
    expect(screen.getByText("Protected page content")).toBeInTheDocument();
    expect(screen.getByText("admin@test.local")).toBeInTheDocument();
  });
});
