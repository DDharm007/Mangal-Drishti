import React, { useEffect, useState } from "react";
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
  const [lastRefreshedAt, setLastRefreshedAt] = useState<string>(() => new Date().toISOString());

  useEffect(() => {
    onRefresh();
    setLastRefreshedAt(new Date().toISOString());
  }, [onRefresh]);

  const handleManualRefresh = () => {
    onRefresh();
    setLastRefreshedAt(new Date().toISOString());
  };

  const apiOnline = !statusError && systemStatus?.status === "online";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#303431] pb-3">
        <div>
          <div className="text-[10px] font-mono-tech uppercase tracking-widest text-[#C87951]">
            Mission Control Infrastructure Diagnostics
          </div>
          <h2 className="text-[17px] font-semibold text-[#EAE9E4] mt-0.5">
            Backend, Compute, Checkpoints &amp; Runtime Status
          </h2>
        </div>

        <button
          type="button"
          onClick={handleManualRefresh}
          className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#202321] hover:bg-[#303431] border border-[#414642] text-[12px] font-mono-tech text-[#EAE9E4] rounded-[2px]"
        >
          <Icon name="rotate-right" className="text-[11px]" />
          <span>Refresh Diagnostics</span>
        </button>
      </div>

      {statusError && (
        <div className="p-3.5 border border-[#C76E64]/50 bg-[#C76E64]/10 text-[12px] text-[#EAE9E4]">
          <Icon name="triangle-exclamation" className="mr-2 text-[#C76E64]" />
          Backend connection error: {statusError}
        </div>
      )}

      {/* Three Organized Telemetry Panels: Backend, Compute, Runtime */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Panel 1: Backend */}
        <section className="mission-panel p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#303431] pb-2">
            <h3 className="text-[12px] font-mono-tech font-semibold uppercase text-[#EAE9E4] flex items-center gap-2">
              <Icon name="server" className="text-[#C87951] text-[11px]" />
              <span>Backend Service</span>
            </h3>
            <span className={`text-[11px] font-mono-tech ${apiOnline ? "text-[#829B79]" : "text-[#C76E64]"}`}>
              {apiOnline ? "Online" : "Offline"}
            </span>
          </div>

          <dl className="space-y-2 text-[11.5px] font-mono-tech">
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">API HEALTH</dt>
              <dd className={apiOnline ? "text-[#829B79]" : "text-[#C76E64]"}>
                {apiOnline ? "Healthy (HTTP 200)" : "Unreachable"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">APPLICATION VERSION</dt>
              <dd className="text-[#EAE9E4]">v{systemStatus?.version ?? "1.0.0"}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">STORED REPORTS</dt>
              <dd className="text-[#EAE9E4]">{systemStatus?.stored_reports_count ?? "—"}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">STORAGE PATH</dt>
              <dd className="text-[#EAE9E4] truncate max-w-[190px]">{systemStatus?.report_storage_dir ?? "—"}</dd>
            </div>
          </dl>
        </section>

        {/* Panel 2: Compute */}
        <section className="mission-panel p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#303431] pb-2">
            <h3 className="text-[12px] font-mono-tech font-semibold uppercase text-[#EAE9E4] flex items-center gap-2">
              <Icon name="microchip" className="text-[#C87951] text-[11px]" />
              <span>Compute Environment</span>
            </h3>
            <span className="text-[11px] font-mono-tech text-[#9B9E98]">
              {systemStatus ? systemStatus.active_device.toUpperCase() : "—"}
            </span>
          </div>

          <dl className="space-y-2 text-[11.5px] font-mono-tech">
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68] shrink-0">GPU MODEL</dt>
              <dd className="text-[#EAE9E4] text-right break-words" title={systemStatus?.gpu_name ?? undefined}>
                {systemStatus?.gpu_name ?? "No CUDA GPU"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">CUDA AVAILABILITY</dt>
              <dd className={systemStatus?.cuda_available ? "text-[#829B79]" : "text-[#D0A064]"}>
                {systemStatus ? (systemStatus.cuda_available ? "Available" : "Unavailable") : "—"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">VRAM / HOST RAM</dt>
              <dd className="text-[#EAE9E4]">
                {systemStatus ? `${systemStatus.gpu_vram_gb ?? 0} GB / ${systemStatus.host_ram_gb} GB` : "—"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">PYTORCH BUILD</dt>
              <dd className="text-[#EAE9E4]">{systemStatus?.pytorch_version ?? "—"}</dd>
            </div>
          </dl>
        </section>

        {/* Panel 3: Runtime */}
        <section className="mission-panel p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#303431] pb-2">
            <h3 className="text-[12px] font-mono-tech font-semibold uppercase text-[#EAE9E4] flex items-center gap-2">
              <Icon name="gauge" className="text-[#C87951] text-[11px]" />
              <span>Runtime State</span>
            </h3>
            <span className="text-[11px] font-mono-tech text-[#9B9E98]">
              {systemStatus ? `${systemStatus.loaded_models.length} Cached` : "—"}
            </span>
          </div>

          <dl className="space-y-2 text-[11.5px] font-mono-tech">
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">LAST ANALYSIS</dt>
              <dd className="text-[#EAE9E4] truncate max-w-[180px]" title={systemStatus?.latest_analysis_id ?? undefined}>
                {systemStatus?.latest_analysis_id ?? "None"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">ANALYSIS TIMESTAMP</dt>
              <dd className="text-[#EAE9E4]">{formatUtcTimestamp(systemStatus?.latest_analysis_timestamp_utc)}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">INFERENCE LATENCY</dt>
              <dd className="text-[#EAE9E4]">
                {systemStatus?.latest_inference_latency_ms != null
                  ? `${systemStatus.latest_inference_latency_ms.toFixed(2)} ms`
                  : "Unavailable (Batch run)"}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-[#696E68]">LAST REFRESH</dt>
              <dd className="text-[#9B9E98]">{formatUtcTimestamp(lastRefreshedAt)}</dd>
            </div>
          </dl>
        </section>
      </div>

      {/* Panel 4: Model Checkpoints Inventory */}
      <section className="mission-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-3 border-b border-[#303431]">
          <h3 className="text-[12.5px] font-mono-tech font-semibold uppercase text-[#EAE9E4]">
            Model Checkpoints &amp; Memory Cache State
          </h3>
          <div className="flex items-center gap-3 text-[10.5px] font-mono-tech text-[#9B9E98]">
            <span><span className="text-[#829B79]">●</span> Cached in Memory</span>
            <span><span className="text-[#EAE9E4]">●</span> Available on Disk</span>
            <span><span className="text-[#C76E64]">●</span> Unavailable</span>
          </div>
        </div>

        {!systemStatus ? (
          <div className="text-[12px] text-[#9B9E98]">Checkpoint inventory unavailable while API is offline.</div>
        ) : (
          <div className="table-wrapper" tabIndex={0} role="region" aria-label="Registered Model Checkpoints">
            <table className="w-full text-left text-[12px] border-collapse">
              <thead>
                <tr className="border-b border-[#303431] bg-[#121414] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                  <th className="py-2 px-3">Model ID</th>
                  <th className="py-2 px-3">Label</th>
                  <th className="py-2 px-3">Checkpoint Path</th>
                  <th className="py-2 px-3">File Size</th>
                  <th className="py-2 px-3">Disk State</th>
                  <th className="py-2 px-3">Runtime Cache</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#303431] font-mono-tech text-[11px]">
                {systemStatus.checkpoints.map((c) => (
                  <tr key={c.model_name} className="hover:bg-[#202321]/60">
                    <td className="py-2.5 px-3 font-semibold text-[#EAE9E4]">{c.model_name}</td>
                    <td className="py-2.5 px-3 font-sans text-[#EAE9E4]">{c.label}</td>
                    <td className="py-2.5 px-3 text-[#9B9E98]">{c.checkpoint_path}</td>
                    <td className="py-2.5 px-3">{c.size_mb !== null ? `${c.size_mb.toFixed(2)} MB` : "—"}</td>
                    <td className="py-2.5 px-3">
                      <span className={c.available ? "text-[#EAE9E4]" : "text-[#C76E64]"}>
                        {c.available ? "Available on disk" : "Unavailable"}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {!c.available ? (
                        <span className="text-[#C76E64]">Unavailable</span>
                      ) : c.loaded_in_memory ? (
                        <span className="text-[#829B79]">Loaded in memory · Cached</span>
                      ) : (
                        <span className="text-[#9B9E98]">Not loaded (Lazy init)</span>
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
