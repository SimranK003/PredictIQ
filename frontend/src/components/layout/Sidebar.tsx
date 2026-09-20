"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

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

  return (
    <nav
      aria-label="Primary"
      className="border-b border-slate-200 bg-white md:h-screen md:w-56 md:shrink-0 md:border-b-0 md:border-r"
    >
      <div className="px-4 py-4 md:py-5">
        <Link href="/dashboard" className="block text-sm font-bold tracking-tight text-slate-900">
          PredictIQ
        </Link>
        <p className="text-xs text-slate-400">ML Operations</p>
      </div>
      <ul className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:overflow-visible md:px-3 md:pb-4">
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
    </nav>
  );
}
