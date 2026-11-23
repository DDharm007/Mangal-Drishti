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
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ systemStatus, statusError }) => {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const telemetryRef = useRef<HTMLDivElement | null>(null);

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

  return (
    <div className="space-y-6">
      {/* Main Scientific Mission Header */}
      <section className="border-b border-[#303431] pb-6 flex flex-col lg:flex-row lg:items-end justify-between gap-5">
        <div className="space-y-2 max-w-3xl">
          <div className="inline-flex items-center gap-2 text-[11px] font-mono-tech uppercase tracking-[0.16em] text-[#C87951]">
            <Icon name="satellite" className="text-[11px]" />
            <span>Planetary Surface Perception &amp; Geospatial Intelligence</span>
          </div>
          <SplitText
            text="MANGAL DRISHTI"
            as="h1"
            className="font-serif-editorial text-3xl sm:text-4xl lg:text-[42px] font-medium tracking-tight text-[#EAE9E4] leading-tight"
          />
          <BlurText
            text="Decoding the Red Planet Through Artificial Intelligence."
            className="text-[15px] sm:text-[16px] text-[#9B9E98]"
          />
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="px-3 py-1.5 bg-[#181A19] border border-[#303431] rounded-[2px] text-[11px] font-mono-tech text-[#9B9E98] flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isOnline ? "bg-[#829B79]" : "bg-[#C76E64]"}`} />
            <span>SYSTEM: {isOnline ? "OPERATIONAL" : "DEGRADED / OFFLINE"}</span>
          </div>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#C87951] hover:bg-[#A65E40] text-[#0D0E0E] font-semibold text-[13px] rounded-[2px] transition-colors"
          >
            <Icon name="crosshairs" className="text-[12px]" />
            <span>New Terrain Analysis</span>
          </Link>
        </div>
      </section>

      {/* Real Backend Telemetry Strip (no fake numbers) */}
      <div
        ref={telemetryRef}
        className="grid grid-cols-2 lg:grid-cols-6 border border-[#303431] bg-[#181A19] divide-y lg:divide-y-0 lg:divide-x divide-[#303431]"
      >
        <div data-telemetry-cell className="p-4">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Stored Analyses</div>
          <div className="text-[22px] font-mono-tech font-semibold text-[#EAE9E4] mt-1">
            {systemStatus ? systemStatus.stored_reports_count : loading ? "—" : reports.length}
          </div>
        </div>
        <div data-telemetry-cell className="p-4">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Available Models</div>
          <div className="text-[22px] font-mono-tech font-semibold text-[#EAE9E4] mt-1">
            {availableModelsCount !== null ? `${availableModelsCount} / ${systemStatus?.checkpoints.length}` : "—"}
          </div>
        </div>
        <div data-telemetry-cell className="p-4">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Inference Device</div>
          <div className="text-[14px] font-mono-tech font-semibold text-[#EAE9E4] mt-1.5 truncate">
            {systemStatus
              ? systemStatus.cuda_available
                ? `CUDA (${systemStatus.gpu_vram_gb ?? "?"} GB)`
                : "CPU"
              : "—"}
          </div>
        </div>
        <div data-telemetry-cell className="p-4">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Backend Status</div>
          <div className={`text-[14px] font-mono-tech font-semibold mt-1.5 ${isOnline ? "text-[#829B79]" : "text-[#C76E64]"}`}>
            {isOnline ? "Connected" : "Unavailable"}
          </div>
        </div>
        <div data-telemetry-cell className="p-4">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Latest Dominant Terrain</div>
          <div className="text-[14px] font-semibold text-[#EAE9E4] mt-1.5 capitalize">
            {featured?.dominant_terrain?.replace("_", " ") ?? "No analyses"}
          </div>
        </div>
        <div data-telemetry-cell className="p-4">
          <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68]">Latest Timestamp</div>
          <div className="text-[11px] font-mono-tech text-[#9B9E98] mt-1.5">
            {formatUtcTimestamp(systemStatus?.latest_analysis_timestamp_utc ?? featured?.generated_at_utc)}
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 border border-[#C76E64]/45 bg-[#C76E64]/10 text-[13px] text-[#EAE9E4] rounded-[3px]">
          <Icon name="triangle-exclamation" className="mr-2 text-[#C76E64]" />
          {error}
        </div>
      )}

      {/* Featured Most Recent Actual Rover Analysis */}
      {featured && (
        <FadeContent>
          <SpotlightSurface className="mission-panel p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-4 border-b border-[#303431]">
              <div>
                <div className="text-[10px] font-mono-tech uppercase tracking-[0.14em] text-[#C87951]">
                  Featured Latest Observation
                </div>
                <h2 className="text-[18px] font-mono-tech font-semibold text-[#EAE9E4] mt-0.5 break-all">
                  {featured.image_id}
                </h2>
              </div>
              <Link
                to={`/analyze?report=${encodeURIComponent(featured.report_id)}`}
                className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#202321] hover:bg-[#303431] border border-[#414642] text-[12px] text-[#EAE9E4] rounded-[2px] transition-colors"
              >
                <span>Open Interactive Inspector</span>
                <Icon name="arrow-right" className="text-[11px]" />
              </Link>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
              <div className="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="border border-[#303431] bg-[#0D0E0E] overflow-hidden">
                  <div className="px-3 py-1.5 border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] flex justify-between">
                    <span>Original Rover Frame</span>
                    <span>512×512</span>
                  </div>
                  {resolveArtifactUrl(featured.artifact_urls.original_image) ? (
                    <img
                      src={resolveArtifactUrl(featured.artifact_urls.original_image)!}
                      alt={`${featured.image_id} original`}
                      className="w-full aspect-square object-contain block"
                    />
                  ) : (
                    <div className="aspect-square flex items-center justify-center text-[12px] text-[#696E68]">
                      Original image unavailable
                    </div>
                  )}
                </div>

                <div className="border border-[#303431] bg-[#0D0E0E] overflow-hidden">
                  <div className="px-3 py-1.5 border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] flex justify-between">
                    <span>Predicted Segmentation Overlay</span>
                    <span>{formatModelName(featured.model)}</span>
                  </div>
                  {resolveArtifactUrl(featured.artifact_urls.terrain_overlay) ? (
                    <img
                      src={resolveArtifactUrl(featured.artifact_urls.terrain_overlay)!}
                      alt={`${featured.image_id} segmentation overlay`}
                      className="w-full aspect-square object-contain block"
                    />
                  ) : (
                    <div className="aspect-square flex items-center justify-center text-[12px] text-[#696E68]">
                      Overlay artifact unavailable
                    </div>
                  )}
                </div>
              </div>

              <div className="lg:col-span-4 space-y-4">
                <dl className="divide-y divide-[#303431] border border-[#303431] bg-[#121414] text-[12px]">
                  <div className="px-3.5 py-2.5 flex justify-between gap-2">
                    <dt className="text-[#9B9E98] font-mono-tech uppercase text-[11px]">Mission</dt>
                    <dd className="text-[#EAE9E4] font-medium text-right">
                      {featured.mission ?? "Not specified / External"}
                    </dd>
                  </div>
                  <div className="px-3.5 py-2.5 flex justify-between gap-2">
                    <dt className="text-[#9B9E98] font-mono-tech uppercase text-[11px]">Model Used</dt>
                    <dd className="text-[#EAE9E4] font-mono-tech text-right">{featured.model}</dd>
                  </div>
                  <div className="px-3.5 py-2.5 flex justify-between gap-2">
                    <dt className="text-[#9B9E98] font-mono-tech uppercase text-[11px]">Dominant Terrain</dt>
                    <dd className="text-[#EAE9E4] capitalize font-semibold text-right">
                      {featured.dominant_terrain?.replace("_", " ") ?? "—"}
                    </dd>
                  </div>
                  <div className="px-3.5 py-2.5 flex justify-between gap-2">
                    <dt className="text-[#9B9E98] font-mono-tech uppercase text-[11px]">Visual Concern</dt>
                    <dd className={`font-semibold text-right ${formatConcernLabel(featured.traversability_assessment).toneClass}`}>
                      {formatConcernLabel(featured.traversability_assessment).label}
                    </dd>
                  </div>
                  <div className="px-3.5 py-2.5 flex justify-between gap-2">
                    <dt className="text-[#9B9E98] font-mono-tech uppercase text-[11px]">Mean Confidence</dt>
                    <dd className="text-[#EAE9E4] font-mono-tech text-right">
                      {featured.mean_confidence !== null ? featured.mean_confidence.toFixed(4) : "—"}
                    </dd>
                  </div>
                </dl>

                <div className="border border-[#303431] bg-[#121414] p-3.5 space-y-2">
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
              </div>
            </div>
          </SpotlightSurface>
        </FadeContent>
      )}

      {/* Recent Stored Reports Table */}
      <section className="mission-panel">
        <div className="px-4 py-3 border-b border-[#303431] flex items-center justify-between gap-2">
          <h2 className="text-[13px] font-semibold tracking-wide uppercase text-[#EAE9E4] flex items-center gap-2">
            <Icon name="clock-rotate-left" className="text-[#C87951] text-[12px]" />
            <span>Recent Terrain Intelligence Reports</span>
          </h2>
          <Link to="/reports" className="text-[12px] text-[#C87951] hover:text-[#EAE9E4] font-mono-tech">
            View All ({reports.length}) →
          </Link>
        </div>

        <div className="table-wrapper" tabIndex={0} role="region" aria-label="Recent Terrain Intelligence Reports">
          <table className="w-full text-left text-[12px] border-collapse">
            <thead>
              <tr className="border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98] bg-[#121414]">
                <th className="py-2.5 px-4">Report Identifier</th>
                <th className="py-2.5 px-3">Mission</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Analysis Date (UTC)</th>
                <th className="py-2.5 px-3">Dominant Terrain</th>
                <th className="py-2.5 px-3">Visual Concern (Exp.)</th>
                <th className="py-2.5 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#303431]/70">
              {loading ? (
                <tr>
                  <td colSpan={7} className="py-6 px-4 text-center text-[#9B9E98] font-mono-tech">
                    Loading mission reports...
                  </td>
                </tr>
              ) : reports.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-6 px-4 text-center text-[#9B9E98]">
                    No stored analyses found under experiments/intelligence/.
                  </td>
                </tr>
              ) : (
                reports.slice(0, 8).map((r) => {
                  const concern = formatConcernLabel(r.traversability_assessment);
                  return (
                    <tr key={r.report_id} className="hover:bg-[#202321]/70 transition-colors">
                      <td className="py-2.5 px-4 font-mono-tech font-medium text-[#EAE9E4]">{r.report_id}</td>
                      <td className="py-2.5 px-3 text-[#9B9E98]">{r.mission ?? "—"}</td>
                      <td className="py-2.5 px-3 font-mono-tech text-[#9B9E98]">{r.model}</td>
                      <td className="py-2.5 px-3 font-mono-tech text-[#696E68]">
                        {formatUtcTimestamp(r.generated_at_utc)}
                      </td>
                      <td className="py-2.5 px-3 capitalize text-[#EAE9E4]">
                        {r.dominant_terrain?.replace("_", " ") ?? "—"}
                      </td>
                      <td className={`py-2.5 px-3 font-medium ${concern.toneClass}`}>{concern.label}</td>
                      <td className="py-2.5 px-4 text-right">
                        <Link
                          to={`/analyze?report=${encodeURIComponent(r.report_id)}`}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 border border-[#303431] hover:border-[#C87951] text-[11px] font-mono-tech text-[#EAE9E4] rounded-[2px] transition-colors"
                        >
                          <span>Open Report</span>
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
