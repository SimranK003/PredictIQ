export function Pagination({
  total,
  limit,
  offset,
  onOffsetChange,
}: {
  total: number;
  limit: number;
  offset: number;
  onOffsetChange: (offset: number) => void;
}) {
  const page = Math.floor(offset / limit) + 1;
  const pageCount = Math.max(1, Math.ceil(total / limit));
  const canPrev = offset > 0;
  const canNext = offset + limit < total;

  return (
    <nav
      aria-label="Pagination"
      className="flex items-center justify-between border-t border-slate-100 pt-3 text-sm text-slate-500"
    >
      <span>
        Page {page} of {pageCount} · {total.toLocaleString()} total
      </span>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={!canPrev}
          onClick={() => onOffsetChange(Math.max(0, offset - limit))}
          className="rounded-md border border-slate-200 px-3 py-1 font-medium text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Previous
        </button>
        <button
          type="button"
          disabled={!canNext}
          onClick={() => onOffsetChange(offset + limit)}
          className="rounded-md border border-slate-200 px-3 py-1 font-medium text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </nav>
  );
}
