"use client";

import { createContext, useCallback, useContext } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import type { AuthUser } from "@/lib/types";

interface AuthContextValue {
  user: AuthUser | null;
  /** True while the initial session check is in flight — including the
   * pending window during a retry's backoff, not just the very first
   * fetch (see this project's own note on isPending vs isLoading in
   * the README's interview talking points: isLoading is only true on
   * the very first fetch, which previously caused a real bug here). */
  isPending: boolean;
  refetchUser: () => Promise<unknown>;
  logout: () => Promise<void>;
}

// Exported so tests can supply a fixed auth state (see
// src/test/queryClientWrapper.tsx's TestAuthProvider) without hitting
// the real /auth/me query and needing every page test's `@/lib/api`
// mock to also stub getMe.
export const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();

  const { data: user, isPending } = useQuery({
    queryKey: queryKeys.currentUser,
    queryFn: async () => {
      try {
        return await api.getMe();
      } catch (error) {
        // 401 just means "not logged in" — a normal, expected state,
        // not a fetch failure worth retrying or surfacing as an error.
        if (error instanceof ApiError && error.status === 401) {
          return null;
        }
        throw error;
      }
    },
    retry: false,
    staleTime: 60_000,
  });

  const refetchUser = useCallback(
    () => queryClient.invalidateQueries({ queryKey: queryKeys.currentUser }),
    [queryClient],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      queryClient.setQueryData(queryKeys.currentUser, null);
      await queryClient.invalidateQueries({ queryKey: queryKeys.currentUser });
    }
  }, [queryClient]);

  return (
    <AuthContext.Provider value={{ user: user ?? null, isPending, refetchUser, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
