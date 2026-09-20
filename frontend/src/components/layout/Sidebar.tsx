"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import clsx from "clsx";
import { useAuth } from "@/components/auth/AuthProvider";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Overview" },
  { href: "/models", label: "Model Registry" },
  { href: "/jobs", label: "Training Jobs" },
  { href: "/predictions", label: "Predictions" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/train", label: "Train" },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [isSigningOut, setIsSigningOut] = useState(false);

  async function handleSignOut() {
    setIsSigningOut(true);
    try {
      await logout();
    } finally {
      setIsSigningOut(false);
      router.replace("/login");
    }
  }

  return (
    <nav
      aria-label="Primary"
      className="flex flex-col border-b border-slate-200 bg-white md:h-screen md:w-56 md:shrink-0 md:border-b-0 md:border-r"
    >
      <div className="px-4 py-4 md:py-5">
        <Link href="/dashboard" className="block text-sm font-bold tracking-tight text-slate-900">
          PredictIQ
        </Link>
        <p className="text-xs text-slate-400">ML Operations</p>
      </div>
      <ul className="flex flex-1 gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:overflow-visible md:px-3 md:pb-4">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href || pathname?.startsWith(`${item.href}/`);
          return (
            <li key={item.href} className="shrink-0 md:shrink">
              <Link
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={clsx(
                  "block whitespace-nowrap rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-indigo-50 text-indigo-700"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900",
                )}
              >
                {item.label}
              </Link>
            </li>
          );
        })}
      </ul>
      {user && (
        <div className="border-t border-slate-200 px-4 py-3">
          <p className="truncate text-xs text-slate-500" title={user.email}>
            {user.email}
            {user.is_admin && <span className="ml-1 text-slate-400">· Admin</span>}
          </p>
          <button
            type="button"
            onClick={handleSignOut}
            disabled={isSigningOut}
            className="mt-1 text-xs font-medium text-indigo-600 hover:text-indigo-500 disabled:opacity-50"
          >
            {isSigningOut ? "Signing out…" : "Sign out"}
          </button>
        </div>
      )}
    </nav>
  );
}
