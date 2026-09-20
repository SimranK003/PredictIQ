"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Sidebar } from "@/components/layout/Sidebar";
import { useAuth } from "./AuthProvider";

const LOGIN_ROUTE = "/login";

function FullPageLoading() {
  return (
    <div
      role="status"
      aria-label="Loading"
      className="flex min-h-screen w-full items-center justify-center bg-slate-50"
    >
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
    </div>
  );
}

/**
 * Client-side route gate: redirects to /login when there's no session,
 * and away from /login once there is one. This is a UX convenience, not
 * the security boundary — every protected API call is independently
 * rejected by the backend (see app/core/security.py's
 * get_current_user/require_admin) regardless of what this component
 * does, since a client-side redirect alone would be trivially
 * bypassable.
 *
 * Renders a spinner (never the sidebar or page content) for the entire
 * window where auth state is unknown or a redirect is about to happen,
 * so there's no flash of protected content before the redirect fires.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isPending } = useAuth();
  const isLoginRoute = pathname === LOGIN_ROUTE;

  const shouldRedirectToLogin = !isPending && !user && !isLoginRoute;
  const shouldRedirectToDashboard = !isPending && !!user && isLoginRoute;

  useEffect(() => {
    if (shouldRedirectToLogin) {
      router.replace(LOGIN_ROUTE);
    } else if (shouldRedirectToDashboard) {
      router.replace("/dashboard");
    }
  }, [shouldRedirectToLogin, shouldRedirectToDashboard, router]);

  if (isPending || shouldRedirectToLogin || shouldRedirectToDashboard) {
    return <FullPageLoading />;
  }

  if (isLoginRoute) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <Sidebar />
      <div id="main-content" className="flex flex-1 flex-col">
        {children}
      </div>
    </div>
  );
}
