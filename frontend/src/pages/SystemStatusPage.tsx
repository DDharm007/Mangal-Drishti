import React, { useEffect } from "react";
import { Icon } from "../components/ui/Icon";
import type { SystemStatusResponse } from "../types/api";
import { formatUtcTimestamp } from "../utils/terrain";

export interface SystemStatusPageProps {
  systemStatus: SystemStatusResponse | null;
  statusError: string | null;
  onRefresh: () => void;
}

export const SystemStatusPage: React.FC<SystemStatusPageProps> = ({
  systemStatus,
  statusError,
  onRefresh,
}) => {
  useEffect(() => {
    onRefresh();
  }, [onRefresh]);

  const apiState = statusError || !systemStatus ? "Offline" : "Online";

  return (
    <div className="space-y-5">
      <section className="mission-panel p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-4 border-b border-[#303431]">
          <div>
            <div className="text-[10px] font-mono-tech uppercase tracking-widest text-[#C87951]">
              Live Infrastructure Telemetry
            </div>
            <h2 className="text-[18px] font-semibold text-[#EAE9E4] mt-0.5">
              Backend, Compute &amp; Checkpoint Diagnostics
            </h2>
          </div>

          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#202321] hover:bg-[#303431] border border-[#414642] text-[12px] font-mono-tech text-[#EAE9E4] rounded-[2px]"
          >
            <Icon name="rotate-right" className="text-[11px]" />
            <span>Refresh Diagnostics</span>
          </button>
        </div>

        {statusError && (
          <div className="mb-4 p-3 border border-[#C76E64]/50 bg-[#C76E64]/10 text-[12px] text-[#EAE9E4]">
            Backend connection failed: {statusError}
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 font-mono-tech text-[12px]">
          <div className="p-3.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">API CONNECTIVITY</div>
            <div
              className={`text-[16px] font-semibold mt-1 flex items-center gap-2 ${
                apiState === "Online" ? "text-[#829B79]" : "text-[#C76E64]"
              }`}
            >
              <Icon name={apiState === "Online" ? "circle-check" : "circle-xmark"} className="text-[14px]" />
              <span>{apiState}</span>
            </div>
          </div>

          <div className="p-3.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">PYTORCH / CUDA</div>
            <div className="text-[#EAE9E4] font-semibold mt-1">
              {systemStatus ? `${systemStatus.pytorch_version}` : "Unavailable"}
            </div>
            <div className="text-[11px] text-[#9B9E98] mt-0.5">
              CUDA: {systemStatus ? (systemStatus.cuda_available ? "Available" : "Unavailable") : "—"}
            </div>
          </div>

          <div className="p-3.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">ACTIVE COMPUTE HARDWARE</div>
            <div className="text-[#EAE9E4] font-semibold mt-1 truncate">
              {systemStatus?.gpu_name ?? (systemStatus ? `CPU (${systemStatus.active_device})` : "Unavailable")}
            </div>
            <div className="text-[11px] text-[#9B9E98] mt-0.5">
              {systemStatus?.gpu_vram_gb ? `${systemStatus.gpu_vram_gb} GB VRAM` : "No CUDA VRAM"} ·{" "}
              {systemStatus?.host_ram_gb ?? "?"} GB RAM
            </div>
          </div>

          <div className="p-3.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">REPORT STORAGE &amp; VERSION</div>
            <div className="text-[#EAE9E4] font-semibold mt-1">
              {systemStatus?.stored_reports_count ?? "—"} stored reports
            </div>
            <div className="text-[11px] text-[#9B9E98] mt-0.5">App v{systemStatus?.version ?? "1.0.0"}</div>
          </div>
        </div>

        {systemStatus && (
          <div className="mt-4 pt-3 border-t border-[#303431] flex flex-wrap gap-6 text-[11px] font-mono-tech text-[#9B9E98]">
            <span>
              STORAGE DIR: <strong className="text-[#EAE9E4]">{systemStatus.report_storage_dir}</strong>
            </span>
            <span>
              LAST ANALYSIS ID: <strong className="text-[#EAE9E4]">{systemStatus.latest_analysis_id ?? "None"}</strong>
            </span>
            <span>
              LAST TIMESTAMP:{" "}
              <strong className="text-[#EAE9E4]">
                {formatUtcTimestamp(systemStatus.latest_analysis_timestamp_utc)}
              </strong>
            </span>
          </div>
        )}
      </section>

      {/* Checkpoints Inventory Table */}
      <section className="mission-panel p-5">
        <h3 className="text-[13px] font-semibold uppercase text-[#EAE9E4] pb-3 mb-3 border-b border-[#303431]">
          Registered Segmentation Checkpoints
        </h3>

        {!systemStatus ? (
          <div className="text-[12px] text-[#9B9E98]">Checkpoint inventory unavailable while API is offline.</div>
        ) : (
          <div className="table-wrapper" tabIndex={0} role="region" aria-label="Registered Model Checkpoints">
            <table className="w-full text-left text-[12px] border-collapse">
              <thead>
                <tr className="border-b border-[#303431] bg-[#121414] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                  <th className="py-2.5 px-3">Model ID</th>
                  <th className="py-2.5 px-3">Label</th>
                  <th className="py-2.5 px-3">Checkpoint Path</th>
                  <th className="py-2.5 px-3">Disk Status</th>
                  <th className="py-2.5 px-3">File Size</th>
                  <th className="py-2.5 px-3">Memory Cache</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#303431] font-mono-tech text-[11px]">
                {systemStatus.checkpoints.map((c) => (
                  <tr key={c.model_name} className="hover:bg-[#202321]/60">
                    <td className="py-2.5 px-3 font-semibold text-[#EAE9E4]">{c.model_name}</td>
                    <td className="py-2.5 px-3 font-sans text-[#EAE9E4]">{c.label}</td>
                    <td className="py-2.5 px-3 text-[#9B9E98]">{c.checkpoint_path}</td>
                    <td className="py-2.5 px-3">
                      <span className={c.available ? "text-[#829B79]" : "text-[#C76E64]"}>
                        {c.available ? "Online / Ready" : "Unavailable"}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">{c.size_mb !== null ? `${c.size_mb.toFixed(2)} MB` : "—"}</td>
                    <td className="py-2.5 px-3">
                      {c.loaded_in_memory ? (
                        <span className="text-[#829B79]">Cached</span>
                      ) : (
                        <span className="text-[#696E68]">Lazy (On-Demand)</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
