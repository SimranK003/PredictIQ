import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { AuthContext } from "@/components/auth/AuthProvider";
import type { AuthUser } from "@/lib/types";

/** A fresh, retry-disabled QueryClient per test — avoids real network
 * retries slowing down failure-path tests, and avoids state leaking
 * between tests. */
export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0, gcTime: 0 },
      mutations: { retry: false },
    },
  });
}

const DEFAULT_TEST_USER: AuthUser = {
  id: "test-admin-id",
  email: "admin@test.local",
  is_admin: true,
  created_at: "2026-01-01T00:00:00Z",
  last_login_at: null,
};

/** Supplies a fixed auth context directly (bypassing the real /auth/me
 * query) so component tests don't need to also stub `api.getMe` in
 * every file's `@/lib/api` mock. Defaults to an admin user, matching
 * this suite's existing pages-render-as-if-authenticated assumption —
 * pass `user={null}` to test an unauthenticated/non-admin render.
 */
export function TestAuthProvider({
  children,
  user = DEFAULT_TEST_USER,
}: {
  children: ReactNode;
  user?: AuthUser | null;
}) {
  return (
    <AuthContext.Provider
      value={{
        user,
        isPending: false,
        refetchUser: async () => undefined,
        logout: async () => undefined,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function TestQueryProvider({
  children,
  user,
}: {
  children: ReactNode;
  user?: AuthUser | null;
}) {
  return (
    <QueryClientProvider client={createTestQueryClient()}>
      <TestAuthProvider user={user}>{children}</TestAuthProvider>
    </QueryClientProvider>
  );
}
