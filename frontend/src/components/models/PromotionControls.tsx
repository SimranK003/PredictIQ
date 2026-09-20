"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { queryKeys } from "@/lib/queryKeys";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { formatPercent } from "@/lib/format";
import type { ModelVersionDetail } from "@/lib/types";

/** Promote/rollback both require an explicit confirmation showing what
 * will change — current production vs. the candidate's own metrics,
 * dataset, and training job — never a one-click destructive action. */
export function PromotionControls({ model }: { model: ModelVersionDetail }) {
  const queryClient = useQueryClient();
  const [promoteOpen, setPromoteOpen] = useState(false);
  const [rollbackOpen, setRollbackOpen] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const productionQuery = useQuery({
    queryKey: queryKeys.productionModel,
    queryFn: api.getProductionModel,
    enabled: promoteOpen,
    retry: false,
  });

  const invalidateAfterLifecycleChange = () => {
    queryClient.invalidateQueries({ queryKey: ["models"] });
  };

  const promoteMutation = useMutation({
    mutationFn: () => api.promoteModel(model.id),
    onSuccess: () => {
      setPromoteOpen(false);
      setActionError(null);
      invalidateAfterLifecycleChange();
    },
    onError: (error: unknown) => {
      setActionError(error instanceof ApiError ? error.message : "Promotion failed.");
    },
  });

  const rollbackMutation = useMutation({
    mutationFn: () => api.rollbackModel(),
    onSuccess: () => {
      setRollbackOpen(false);
      setActionError(null);
      invalidateAfterLifecycleChange();
    },
    onError: (error: unknown) => {
      setActionError(error instanceof ApiError ? error.message : "Rollback failed.");
    },
  });

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        {model.stage === "candidate" && (
          <button
            type="button"
            onClick={() => setPromoteOpen(true)}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Promote to Production
          </button>
        )}
        {model.stage === "production" && (
          <button
            type="button"
            onClick={() => setRollbackOpen(true)}
            className="rounded-md border border-amber-300 bg-amber-50 px-4 py-2 text-sm font-medium text-amber-800 hover:bg-amber-100"
          >
            Roll back to previous
          </button>
        )}
      </div>
      {actionError && <p className="text-sm text-red-600">{actionError}</p>}

      <ConfirmDialog
        open={promoteOpen}
        title={`Promote ${model.version_label} to production?`}
        description="This will demote the current production model to previous and archive whatever was already previous."
        confirmLabel="Promote"
        busy={promoteMutation.isPending}
        onConfirm={() => promoteMutation.mutate()}
        onCancel={() => setPromoteOpen(false)}
      >
        <div className="grid grid-cols-2 gap-4 rounded-md bg-slate-50 p-3 text-xs">
          <div>
            <p className="font-semibold text-slate-500">Current production</p>
            {productionQuery.isPending && <p className="text-slate-400">Loading…</p>}
            {productionQuery.isError && <p className="text-slate-400">None currently.</p>}
            {productionQuery.data && (
              <>
                <p className="text-slate-800">{productionQuery.data.version_label}</p>
                <p className="text-slate-500">ROC-AUC {formatPercent(productionQuery.data.metrics.roc_auc)}</p>
              </>
            )}
          </div>
          <div>
            <p className="font-semibold text-slate-500">Candidate</p>
            <p className="text-slate-800">{model.version_label}</p>
            <p className="text-slate-500">ROC-AUC {formatPercent(model.metrics.roc_auc)}</p>
            <p className="mt-1 text-slate-400">Dataset {model.dataset_id.slice(0, 8)}</p>
          </div>
        </div>
      </ConfirmDialog>

      <ConfirmDialog
        open={rollbackOpen}
        title="Roll back to the previous production model?"
        description={`${model.version_label} will move to "previous" and whatever is currently "previous" will become production again.`}
        confirmLabel="Roll back"
        confirmTone="danger"
        busy={rollbackMutation.isPending}
        onConfirm={() => rollbackMutation.mutate()}
        onCancel={() => setRollbackOpen(false)}
      />
    </div>
  );
}
