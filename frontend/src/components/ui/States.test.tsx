import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EmptyState, ErrorState, LoadingState } from "./States";

describe("LoadingState", () => {
  it("announces itself to assistive tech via role=status", () => {
    render(<LoadingState label="Loading models…" />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading models…");
  });
});

describe("ErrorState", () => {
  it("renders the error message with role=alert", () => {
    render(<ErrorState message="Could not reach the API." />);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not reach the API.");
  });

  it("calls onRetry when the retry button is clicked", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<ErrorState message="Failed to load." onRetry={onRetry} />);

    await user.click(screen.getByRole("button", { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("does not render a retry button when onRetry is not provided", () => {
    render(<ErrorState message="Failed to load." />);
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });
});

describe("EmptyState", () => {
  it("renders the title and optional description", () => {
    render(<EmptyState title="No production model available" description="Promote a candidate." />);
    expect(screen.getByText("No production model available")).toBeInTheDocument();
    expect(screen.getByText("Promote a candidate.")).toBeInTheDocument();
  });
});
