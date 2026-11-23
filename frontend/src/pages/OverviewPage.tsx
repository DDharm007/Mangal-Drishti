import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { BlurText } from "../components/motion/BlurText";
import { FadeContent } from "../components/motion/FadeContent";
import { SplitText } from "../components/motion/SplitText";
import { SpotlightSurface } from "../components/motion/SpotlightSurface";
import { useAnimeStagger } from "../components/motion/useAnimeEntrance";
import { Icon } from "../components/ui/Icon";
import { fetchReports, resolveArtifactUrl } from "../services/api";
import type { ReportSummary, SystemStatusResponse } from "../types/api";
import { TERRAIN_CLASSES, formatConcernLabel, formatModelName, formatUtcTimestamp } from "../utils/terrain";

export interface OverviewPageProps {
  systemStatus: SystemStatusResponse | null;
  statusError: string | null;
  onRefreshStatus?: () => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  systemStatus,
  statusError,
  onRefreshStatus,
}) => {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [obsMode, setObsMode] = useState<"dual" | "overlay" | "compare">("dual");
  const [splitPct, setSplitPct] = useState<number>(50);
  const telemetryRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    onRefreshStatus?.();
  }, [onRefreshStatus]);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        setLoading(true);
        const res = await fetchReports();
        if (active) {
          setReports(res.reports);
          setError(null);
        }
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : "Failed to load mission reports.");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  useAnimeStagger(telemetryRef, "[data-telemetry-cell]", reports.length);

  const featured = reports[0] ?? null;
  const availableModelsCount = systemStatus
    ? systemStatus.checkpoints.filter((c) => c.available).length
    : null;
  const isOnline = systemStatus?.status === "online" && !statusError;

  const origUrl = resolveArtifactUrl(featured?.artifact_urls.original_image);
  const overlayUrl = resolveArtifactUrl(featured?.artifact_urls.terrain_overlay);

  return (
    <div className="space-y-4">
      {/* Compact Mission Identity Strip */}
      <section className="border-b border-[#303431] pb-4 flex flex-col lg:flex-row lg:items-end justify-between gap-3">
        <div className="space-y-1 max-w-3xl">
          <div className="inline-flex items-center gap-2 text-[10px] font-mono-tech uppercase tracking-[0.16em] text-[#C87951]">
            <Icon name="satellite" className="text-[10px]" />
            <span>Planetary Surface Perception &amp; Geospatial Intelligence</span>
          </div>
          <SplitText
            text="MANGAL DRISHTI"
            as="h2"
            className="font-serif-editorial text-2xl sm:text-3xl lg:text-[34px] font-medium tracking-tight text-[#EAE9E4] leading-none"
          />
          <BlurText
            text="Decoding the Red Planet Through Artificial Intelligence."
            className="text-[13.5px] sm:text-[14px] text-[#9B9E98]"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <div className="px-2.5 py-1.5 bg-[#181A19] border border-[#303431] rounded-[2px] text-[10.5px] font-mono-tech text-[#9B9E98] flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? "bg-[#829B79]" : "bg-[#C76E64]"}`} />
            <span>STATUS: {isOnline ? "OPERATIONAL" : "DEGRADED / OFFLINE"}</span>
          </div>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-[#C87951] hover:bg-[#A65E40] text-[#0D0E0E] font-semibold text-[12px] rounded-[2px] transition-colors focus-visible:outline-2 focus-visible:outline-[#EAE9E4]"
          >
            <Icon name="crosshairs" className="text-[11px]" />
            <span>New Terrain Analysis</span>
          </Link>
        </div>
      </section>

      {/* Row 1: Compact 6-Cell Operational Telemetry Bar */}
      <div
        ref={telemetryRef}
        className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 border border-[#303431] bg-[#181A19] divide-y sm:divide-y-0 sm:divide-x divide-[#303431]"
      >
        <div data-telemetry-cell className="px-3.5 py-2.5">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Stored Analyses</div>
          <div className="text-[18px] font-mono-tech font-semibold text-[#EAE9E4] mt-0.5">
            {systemStatus ? systemStatus.stored_reports_count : loading ? "—" : reports.length}
          </div>
        </div>
        <div data-telemetry-cell className="px-3.5 py-2.5">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Available Models</div>
          <div className="text-[18px] font-mono-tech font-semibold text-[#EAE9E4] mt-0.5">
            {availableModelsCount !== null ? `${availableModelsCount} / ${systemStatus?.checkpoints.length}` : "—"}
          </div>
        </div>
        <div data-telemetry-cell className="px-3.5 py-2.5">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Active Device</div>
          <div className="text-[13px] font-mono-tech font-semibold text-[#EAE9E4] mt-1 truncate">
            {systemStatus
              ? systemStatus.cuda_available
                ? `CUDA (${systemStatus.gpu_vram_gb ?? "?"} GB)`
                : "CPU"
              : "—"}
          </div>
        </div>
        <div data-telemetry-cell className="px-3.5 py-2.5">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Backend Status</div>
          <div className={`text-[13px] font-mono-tech font-semibold mt-1 ${isOnline ? "text-[#829B79]" : "text-[#C76E64]"}`}>
            {isOnline ? "Connected" : "Unavailable"}
          </div>
        </div>
        <div data-telemetry-cell className="px-3.5 py-2.5">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Latest Dominant</div>
          <div className="text-[13px] font-semibold text-[#EAE9E4] mt-1 capitalize truncate">
            {featured?.dominant_terrain?.replace("_", " ") ?? "No analyses"}
          </div>
        </div>
        <div data-telemetry-cell className="px-3.5 py-2.5">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Latest Timestamp</div>
          <div className="text-[11px] font-mono-tech text-[#9B9E98] mt-1 truncate">
            {formatUtcTimestamp(systemStatus?.latest_analysis_timestamp_utc ?? featured?.generated_at_utc)}
          </div>
        </div>
      </div>

      {error && (
        <div className="p-3.5 border border-[#C76E64]/45 bg-[#C76E64]/10 text-[12.5px] text-[#EAE9E4] rounded-[2px]">
          <Icon name="triangle-exclamation" className="mr-2 text-[#C76E64]" />
          {error}
        </div>
      )}

      {/* Row 2: Dense 12-Column OrbitTrack-Inspired Observation + Side Operational Panels */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
        {/* Centerpiece Latest Mars Observation (8 cols) */}
        <div className="xl:col-span-8">
          {featured ? (
            <FadeContent>
              <SpotlightSurface className="mission-panel p-4">
                <div className="flex flex-wrap items-center justify-between gap-2.5 pb-3 mb-3 border-b border-[#303431]">
                  <div className="min-w-0">
                    <div className="text-[10px] font-mono-tech uppercase tracking-[0.14em] text-[#C87951]">
                      Latest Mars Rover Observation
                    </div>
                    <h3 className="text-[16px] font-mono-tech font-semibold text-[#EAE9E4] mt-0.5 truncate">
                      {featured.image_id}
                    </h3>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <div className="inline-flex border border-[#303431] bg-[#121414] rounded-[2px] p-0.5 text-[10.5px] font-mono-tech">
                      {(
                        [
                          { id: "dual", label: "Side-by-Side" },
                          { id: "overlay", label: "Overlay" },
                          { id: "compare", label: "Compare Split" },
                        ] as const
                      ).map((m) => (
                        <button
                          key={m.id}
                          type="button"
                          onClick={() => setObsMode(m.id)}
                          className={`px-2 py-1 rounded-[1px] transition-colors ${
                            obsMode === m.id ? "bg-[#202321] text-[#EAE9E4]" : "text-[#9B9E98] hover:text-[#EAE9E4]"
                          }`}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>

                    <Link
                      to={`/analyze?report=${encodeURIComponent(featured.report_id)}`}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-[#202321] hover:bg-[#303431] border border-[#414642] text-[11px] font-mono-tech text-[#EAE9E4] rounded-[2px] transition-colors"
                    >
                      <span>Open Inspector</span>
                      <Icon name="arrow-right" className="text-[10px]" />
                    </Link>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
                  {/* Large Rover Frame View (7/12 of card) */}
                  <div className="lg:col-span-7">
                    {obsMode === "dual" ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                        <div className="border border-[#303431] bg-[#0D0E0E]">
                          <div className="px-2.5 py-1 border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] flex justify-between">
                            <span>Original Frame</span>
                            <span>512×512</span>
                          </div>
                          {origUrl ? (
                            <img src={origUrl} alt={`${featured.image_id} original`} className="w-full aspect-square object-contain block" />
                          ) : (
                            <div className="aspect-square flex items-center justify-center text-[11px] text-[#696E68]">Unavailable</div>
                          )}
                        </div>
                        <div className="border border-[#303431] bg-[#0D0E0E]">
                          <div className="px-2.5 py-1 border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] flex justify-between">
                            <span>Segmentation Overlay</span>
                            <span>{featured.model}</span>
                          </div>
                          {overlayUrl ? (
                            <img src={overlayUrl} alt={`${featured.image_id} overlay`} className="w-full aspect-square object-contain block" />
                          ) : (
                            <div className="aspect-square flex items-center justify-center text-[11px] text-[#696E68]">Unavailable</div>
                          )}
                        </div>
                      </div>
                    ) : obsMode === "overlay" ? (
                      <div className="border border-[#303431] bg-[#0D0E0E]">
                        <div className="px-2.5 py-1 border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] flex justify-between">
                          <span>Predicted Segmentation Overlay</span>
                          <span>{formatModelName(featured.model)}</span>
                        </div>
                        {overlayUrl ? (
                          <img src={overlayUrl} alt={`${featured.image_id} overlay`} className="w-full max-h-[380px] object-contain mx-auto block" />
                        ) : (
                          <div className="aspect-square flex items-center justify-center text-[11px] text-[#696E68]">Unavailable</div>
                        )}
                      </div>
                    ) : (
                      <div className="border border-[#303431] bg-[#0D0E0E]">
                        <div className="px-2.5 py-1 border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] flex items-center justify-between gap-2">
                          <span>Split Comparison ({splitPct}%)</span>
                          <input
                            type="range"
                            min={10}
                            max={90}
                            value={splitPct}
                            onChange={(e) => setSplitPct(Number(e.target.value))}
                            className="w-28"
                            aria-label="Overview split comparison slider"
                          />
                        </div>
                        {origUrl && overlayUrl ? (
                          <div className="relative w-full max-w-[380px] aspect-square mx-auto">
                            <img src={origUrl} alt="Original" className="w-full h-full object-contain block" />
                            <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 0 0 ${splitPct}%)` }}>
                              <img src={overlayUrl} alt="Overlay" className="w-full h-full object-contain block" />
                            </div>
                            <div className="absolute top-0 bottom-0 w-px bg-[#C87951]" style={{ left: `${splitPct}%` }} />
                          </div>
                        ) : (
                          <div className="aspect-square flex items-center justify-center text-[11px] text-[#696E68]">Comparison unavailable</div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Metadata & Composition Breakdown (5/12 of card) */}
                  <div className="lg:col-span-5 space-y-3">
                    <dl className="divide-y divide-[#303431] border border-[#303431] bg-[#121414] text-[11.5px]">
                      <div className="px-3 py-2 flex justify-between gap-2">
                        <dt className="text-[#9B9E98] font-mono-tech uppercase text-[10px]">Mission</dt>
                        <dd className="text-[#EAE9E4] font-medium text-right">{featured.mission ?? "External / Uncatalogued"}</dd>
                      </div>
                      <div className="px-3 py-2 flex justify-between gap-2">
                        <dt className="text-[#9B9E98] font-mono-tech uppercase text-[10px]">Model Used</dt>
                        <dd className="text-[#EAE9E4] font-mono-tech text-right">{featured.model}</dd>
                      </div>
                      <div className="px-3 py-2 flex justify-between gap-2">
                        <dt className="text-[#9B9E98] font-mono-tech uppercase text-[10px]">Dominant Terrain</dt>
                        <dd className="text-[#EAE9E4] capitalize font-semibold text-right">
                          {featured.dominant_terrain?.replace("_", " ") ?? "—"}
                        </dd>
                      </div>
                      <div className="px-3 py-2 flex justify-between gap-2">
                        <dt className="text-[#9B9E98] font-mono-tech uppercase text-[10px]">Visual Concern</dt>
                        <dd className={`font-semibold text-right ${formatConcernLabel(featured.traversability_assessment).toneClass}`}>
                          {formatConcernLabel(featured.traversability_assessment).label}
                        </dd>
                      </div>
                      <div className="px-3 py-2 flex justify-between gap-2">
                        <dt className="text-[#9B9E98] font-mono-tech uppercase text-[10px]">Mean Conf / Valid</dt>
                        <dd className="text-[#EAE9E4] font-mono-tech text-right">
                          {featured.mean_confidence !== null ? featured.mean_confidence.toFixed(4) : "—"} ·{" "}
                          {featured.valid_pixel_percentage?.toFixed(1) ?? "—"}%
                        </dd>
                      </div>
                    </dl>

                    <div className="border border-[#303431] bg-[#121414] p-3 space-y-2">
                      <div className="text-[10px] font-mono-tech uppercase text-[#9B9E98]">Class Coverage Breakdown</div>
                      {TERRAIN_CLASSES.map((c) => {
                        const pct = featured.terrain_summary?.[c.key] ?? 0;
                        return (
                          <div key={c.key} className="flex items-center gap-2 text-[11px]">
                            <span className="w-2 h-2 rounded-[1px]" style={{ backgroundColor: c.hex }} />
                            <span className="w-16 text-[#9B9E98]">{c.label}</span>
                            <div className="flex-1 h-1.5 bg-[#181A19]">
                              <div className="h-full" style={{ width: `${Math.min(100, pct)}%`, backgroundColor: c.hex }} />
                            </div>
                            <span className="font-mono-tech text-[#EAE9E4] w-12 text-right">{pct.toFixed(1)}%</span>
                          </div>
                        );
                      })}
                    </div>

                    <div className="px-3 py-1.5 bg-[#121414] border border-[#C76E64]/35 text-[10.5px] font-mono-tech text-[#C76E64] flex items-center justify-between">
                      <span>Physical safety verified: NO</span>
                      <span className="text-[#696E68]">2D visual only</span>
                    </div>
                  </div>
                </div>
              </SpotlightSurface>
            </FadeContent>
          ) : (
            <div className="mission-panel p-8 text-center text-[13px] text-[#9B9E98]">
              {loading ? "Loading latest Mars observation..." : "No stored analyses available yet."}
            </div>
          )}
        </div>

        {/* Simultaneous Right Operational Stack (4 cols) */}
        <div className="xl:col-span-4 space-y-4">
          {/* System Health & Compute Telemetry Panel */}
          <section className="mission-panel p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-[#303431] pb-2">
              <h3 className="text-[11.5px] font-mono-tech font-semibold uppercase tracking-wider text-[#EAE9E4] flex items-center gap-2">
                <Icon name="microchip" className="text-[#C87951] text-[11px]" />
                <span>System Health &amp; Compute</span>
              </h3>
              <Link to="/status" className="text-[10.5px] font-mono-tech text-[#C87951] hover:text-[#EAE9E4]">
                Diagnostics →
              </Link>
            </div>

            {systemStatus ? (
              <dl className="space-y-1.5 text-[11px] font-mono-tech">
                <div className="flex justify-between gap-2">
                  <dt className="text-[#696E68]">GPU HARDWARE</dt>
                  <dd className="text-[#EAE9E4] text-right truncate max-w-[200px]">
                    {systemStatus.gpu_name?.replace("NVIDIA GeForce ", "") ?? "CPU Only"}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[#696E68]">CUDA / PYTORCH</dt>
                  <dd className="text-[#EAE9E4]">
                    {systemStatus.cuda_available ? "Enabled" : "Disabled"} · {systemStatus.pytorch_version}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[#696E68]">MEMORY (VRAM / RAM)</dt>
                  <dd className="text-[#EAE9E4]">
                    {systemStatus.gpu_vram_gb ?? 0} GB / {systemStatus.host_ram_gb} GB
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[#696E68]">CACHED IN MEMORY</dt>
                  <dd className="text-[#EAE9E4]">
                    {systemStatus.loaded_models.length > 0 ? systemStatus.loaded_models.join(", ") : "Lazy (On-demand)"}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-[#696E68]">LAST INFERENCE</dt>
                  <dd className="text-[#EAE9E4]">
                    {systemStatus.latest_inference_latency_ms != null
                      ? `${systemStatus.latest_inference_latency_ms.toFixed(1)} ms`
                      : "N/A (Batch/Pre-computed)"}
                  </dd>
                </div>
              </dl>
            ) : (
              <div className="text-[11px] font-mono-tech text-[#9B9E98]">Waiting for backend telemetry...</div>
            )}
          </section>

          {/* Model Availability Panel */}
          <section className="mission-panel p-4 space-y-2.5">
            <div className="flex items-center justify-between border-b border-[#303431] pb-2">
              <h3 className="text-[11.5px] font-mono-tech font-semibold uppercase tracking-wider text-[#EAE9E4] flex items-center gap-2">
                <Icon name="layer-group" className="text-[#C87951] text-[11px]" />
                <span>Model Checkpoints</span>
              </h3>
              <Link to="/models" className="text-[10.5px] font-mono-tech text-[#C87951] hover:text-[#EAE9E4]">
                Benchmarks →
              </Link>
            </div>

            {systemStatus ? (
              <div className="divide-y divide-[#303431]/70 text-[11px] font-mono-tech">
                {systemStatus.checkpoints.map((c) => (
                  <div key={c.model_name} className="py-2 flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="font-sans font-medium text-[12px] text-[#EAE9E4] truncate">{c.label}</div>
                      <div className="text-[10px] text-[#696E68]">
                        {c.size_mb !== null ? `${c.size_mb.toFixed(1)} MB` : "Missing"}
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {c.loaded_in_memory && (
                        <span className="px-1.5 py-0.5 bg-[#829B79]/15 border border-[#829B79]/35 text-[#829B79] text-[9.5px] rounded-[2px]">
                          CACHED
                        </span>
                      )}
                      <span
                        className={`px-1.5 py-0.5 border text-[9.5px] rounded-[2px] ${
                          c.available
                            ? "bg-[#202321] border-[#303431] text-[#EAE9E4]"
                            : "bg-[#C76E64]/10 border-[#C76E64]/40 text-[#C76E64]"
                        }`}
                      >
                        {c.available ? "DISK OK" : "MISSING"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-[11px] font-mono-tech text-[#9B9E98]">Unavailable offline.</div>
            )}
          </section>
        </div>
      </div>

      {/* Row 3: Compact Recent Reports Table */}
      <section className="mission-panel">
        <div className="px-4 py-2.5 border-b border-[#303431] flex items-center justify-between gap-2">
          <h2 className="text-[12px] font-mono-tech font-semibold tracking-wider uppercase text-[#EAE9E4] flex items-center gap-2">
            <Icon name="clock-rotate-left" className="text-[#C87951] text-[11px]" />
            <span>Recent Terrain Intelligence Reports</span>
          </h2>
          <Link to="/reports" className="text-[11px] text-[#C87951] hover:text-[#EAE9E4] font-mono-tech">
            Full Archive ({reports.length}) →
          </Link>
        </div>

        <div className="table-wrapper" tabIndex={0} role="region" aria-label="Recent Terrain Intelligence Reports">
          <table className="w-full text-left text-[12px] border-collapse">
            <thead>
              <tr className="border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] bg-[#121414]">
                <th className="py-2 px-4">Report Identifier</th>
                <th className="py-2 px-3">Mission</th>
                <th className="py-2 px-3">Model</th>
                <th className="py-2 px-3">Analysis Date (UTC)</th>
                <th className="py-2 px-3">Dominant</th>
                <th className="py-2 px-3">Visual Concern (Exp.)</th>
                <th className="py-2 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#303431]/70">
              {loading ? (
                <tr>
                  <td colSpan={7} className="py-5 px-4 text-center text-[#9B9E98] font-mono-tech">
                    Loading mission reports...
                  </td>
                </tr>
              ) : reports.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-5 px-4 text-center text-[#9B9E98]">
                    No stored analyses found under experiments/intelligence/.
                  </td>
                </tr>
              ) : (
                reports.slice(0, 8).map((r) => {
                  const concern = formatConcernLabel(r.traversability_assessment);
                  return (
                    <tr key={r.report_id} className="hover:bg-[#202321]/70 transition-colors">
                      <td className="py-2 px-4 font-mono-tech font-medium text-[#EAE9E4]">{r.report_id}</td>
                      <td className="py-2 px-3 text-[#9B9E98]">{r.mission ?? "—"}</td>
                      <td className="py-2 px-3 font-mono-tech text-[#9B9E98]">{r.model}</td>
                      <td className="py-2 px-3 font-mono-tech text-[#696E68]">
                        {formatUtcTimestamp(r.generated_at_utc)}
                      </td>
                      <td className="py-2 px-3 capitalize text-[#EAE9E4]">
                        {r.dominant_terrain?.replace("_", " ") ?? "—"}
                      </td>
                      <td className={`py-2 px-3 font-medium ${concern.toneClass}`}>{concern.label}</td>
                      <td className="py-2 px-4 text-right">
                        <Link
                          to={`/analyze?report=${encodeURIComponent(r.report_id)}`}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 border border-[#303431] hover:border-[#C87951] text-[11px] font-mono-tech text-[#EAE9E4] rounded-[2px] transition-colors"
                        >
                          <span>Open</span>
                          <Icon name="arrow-up-right-from-square" className="text-[10px]" />
                        </Link>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
