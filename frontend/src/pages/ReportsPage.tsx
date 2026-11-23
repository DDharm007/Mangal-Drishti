import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Icon } from "../components/ui/Icon";
import {
  fetchReportDetail,
  fetchReports,
  getDownloadReportUrl,
  resolveArtifactUrl,
} from "../services/api";
import type { ReportDetailResponse, ReportSummary } from "../types/api";
import { TERRAIN_CLASSES, formatConcernLabel, formatUtcTimestamp } from "../utils/terrain";

const CONCERN_FILTERS = [
  { id: "all", label: "All Concern Levels" },
  { id: "lower_visual_concern", label: "Lower Visual Concern" },
  { id: "moderate_visual_concern", label: "Moderate Visual Concern" },
  { id: "higher_visual_concern", label: "Higher Visual Concern" },
  { id: "insufficient_visual_evidence", label: "Insufficient Visual Evidence" },
];

export const ReportsPage: React.FC = () => {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [modelFilter, setModelFilter] = useState("all");
  const [missionFilter, setMissionFilter] = useState("all");
  const [concernFilter, setConcernFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"date_desc" | "date_asc" | "concern_desc">("date_desc");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [jsonModalDetail, setJsonModalDetail] = useState<ReportDetailResponse | null>(null);
  const dialogRef = useRef<HTMLDialogElement | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        setLoading(true);
        const res = await fetchReports();
        if (active) setReports(res.reports);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : "Failed to fetch reports.");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const modelsAvailable = useMemo(
    () => Array.from(new Set(reports.map((r) => r.model))).sort(),
    [reports]
  );
  const missionsAvailable = useMemo(
    () => Array.from(new Set(reports.map((r) => r.mission ?? "Uncatalogued"))).sort(),
    [reports]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = reports.filter((r) => {
      if (q && !r.report_id.toLowerCase().includes(q) && !r.image_id.toLowerCase().includes(q)) {
        return false;
      }
      if (modelFilter !== "all" && r.model !== modelFilter) return false;
      const mLabel = r.mission ?? "Uncatalogued";
      if (missionFilter !== "all" && mLabel !== missionFilter) return false;
      if (concernFilter !== "all" && r.traversability_assessment !== concernFilter) return false;
      return true;
    });

    list.sort((a, b) => {
      if (sortBy === "concern_desc") {
        return (b.experimental_visual_concern_index ?? 0) - (a.experimental_visual_concern_index ?? 0);
      }
      const ta = String(a.generated_at_utc ?? "");
      const tb = String(b.generated_at_utc ?? "");
      return sortBy === "date_asc" ? ta.localeCompare(tb) : tb.localeCompare(ta);
    });
    return list;
  }, [reports, query, modelFilter, missionFilter, concernFilter, sortBy]);

  const handleCopyId = async (reportId: string) => {
    try {
      await navigator.clipboard.writeText(reportId);
      setCopiedId(reportId);
      window.setTimeout(() => setCopiedId((prev) => (prev === reportId ? null : prev)), 1500);
    } catch {
      // Clipboard API unavailable in restricted contexts
    }
  };

  const openJsonInspector = async (reportId: string) => {
    try {
      const det = await fetchReportDetail(reportId);
      setJsonModalDetail(det);
      dialogRef.current?.showModal();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load JSON payload.");
    }
  };

  return (
    <div className="space-y-4">
      {/* Dense Search & Filtering Header */}
      <section className="mission-panel p-3.5 grid grid-cols-1 md:grid-cols-12 gap-3 items-end">
        <div className="md:col-span-4">
          <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1">
            Search Report or Image ID
          </label>
          <div className="relative">
            <Icon name="magnifying-glass" className="absolute left-3 top-2.5 text-[#696E68] text-[11px]" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by ID..."
              className="w-full pl-8 pr-3 py-1.5 bg-[#121414] border border-[#303431] text-[12px] font-mono-tech text-[#EAE9E4] rounded-[2px] focus:border-[#C87951] focus:outline-none"
            />
          </div>
        </div>

        <div className="md:col-span-2">
          <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1">Model</label>
          <select
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
            className="w-full pl-2.5 pr-8 py-1.5 bg-[#121414] border border-[#303431] text-[12px] text-[#EAE9E4] rounded-[2px] truncate"
          >
            <option value="all">All Models</option>
            {modelsAvailable.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1">Mission</label>
          <select
            value={missionFilter}
            onChange={(e) => setMissionFilter(e.target.value)}
            className="w-full pl-2.5 pr-8 py-1.5 bg-[#121414] border border-[#303431] text-[12px] text-[#EAE9E4] rounded-[2px] truncate"
          >
            <option value="all">All Missions</option>
            {missionsAvailable.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1">Visual Concern</label>
          <select
            value={concernFilter}
            onChange={(e) => setConcernFilter(e.target.value)}
            className="w-full pl-2.5 pr-8 py-1.5 bg-[#121414] border border-[#303431] text-[12px] text-[#EAE9E4] rounded-[2px] truncate"
          >
            {CONCERN_FILTERS.map((cf) => (
              <option key={cf.id} value={cf.id}>
                {cf.label}
              </option>
            ))}
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1">Sort Order</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as "date_desc" | "date_asc" | "concern_desc")}
            className="w-full pl-2.5 pr-8 py-1.5 bg-[#121414] border border-[#303431] text-[12px] text-[#EAE9E4] rounded-[2px]"
          >
            <option value="date_desc">Newest First</option>
            <option value="date_asc">Oldest First</option>
            <option value="concern_desc">Concern Index</option>
          </select>
        </div>
      </section>

      {error && (
        <div className="p-3.5 border border-[#C76E64]/50 bg-[#C76E64]/10 text-[12px] text-[#EAE9E4]">
          {error}
        </div>
      )}

      {/* Dense Operational Report Grid */}
      {loading ? (
        <div className="mission-panel p-10 text-center font-mono-tech text-[13px] text-[#9B9E98]">
          Loading reports from experiments/intelligence/...
        </div>
      ) : filtered.length === 0 ? (
        <div className="mission-panel p-10 text-center text-[13px] text-[#9B9E98]">
          No intelligence reports match the active search or filter criteria.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-3.5">
          {filtered.map((r) => {
            const thumb =
              resolveArtifactUrl(r.artifact_urls.terrain_overlay) ??
              resolveArtifactUrl(r.artifact_urls.original_image);
            const concern = formatConcernLabel(r.traversability_assessment);
            const copied = copiedId === r.report_id;
            return (
              <article key={r.report_id} className="mission-panel flex flex-col overflow-hidden hover:border-[#414642] transition-colors">
                <div className="px-3.5 py-2.5 border-b border-[#303431] flex items-start justify-between gap-2 bg-[#121414]">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[12px] font-mono-tech font-semibold text-[#EAE9E4] truncate" title={r.report_id}>
                        {r.report_id}
                      </span>
                      <button
                        type="button"
                        onClick={() => void handleCopyId(r.report_id)}
                        className="p-1 text-[#696E68] hover:text-[#EAE9E4] shrink-0"
                        title="Copy report identifier"
                        aria-label={`Copy report ID ${r.report_id}`}
                      >
                        <Icon name={copied ? "check" : "copy"} className={`text-[10px] ${copied ? "text-[#829B79]" : ""}`} />
                      </button>
                    </div>
                    <div className="text-[11px] text-[#9B9E98]">
                      {r.mission ?? "External / Uncatalogued"} · <span className="font-mono-tech">{r.model}</span>
                    </div>
                  </div>
                  <span
                    className={`shrink-0 px-2 py-0.5 text-[10px] font-mono-tech border rounded-[2px] ${concern.borderClass} ${concern.toneClass}`}
                  >
                    {concern.label}
                  </span>
                </div>

                <div className="grid grid-cols-12 gap-3 p-3.5 flex-1 items-start">
                  <div className="col-span-4 bg-[#0D0E0E] border border-[#303431] aspect-square flex items-center justify-center overflow-hidden">
                    {thumb ? (
                      <img src={thumb} alt={r.report_id} className="w-full h-full object-cover block" />
                    ) : (
                      <span className="text-[10px] font-mono-tech text-[#696E68] px-2 text-center">
                        Unavailable
                      </span>
                    )}
                  </div>

                  <div className="col-span-8 flex flex-col justify-between text-[11px] space-y-2">
                    <div className="space-y-1 font-mono-tech">
                      <div className="flex justify-between text-[#9B9E98]">
                        <span>DOMINANT</span>
                        <span className="text-[#EAE9E4] capitalize">
                          {r.dominant_terrain?.replace("_", " ") ?? "—"}
                        </span>
                      </div>
                      <div className="flex justify-between text-[#9B9E98]">
                        <span>VALID PIXELS</span>
                        <span className="text-[#EAE9E4]">{r.valid_pixel_percentage?.toFixed(1) ?? "—"}%</span>
                      </div>
                      <div className="flex justify-between text-[#9B9E98]">
                        <span>REGIONS / ROCK CANDS</span>
                        <span className="text-[#EAE9E4]">
                          {r.retained_regions} / {r.obstacle_candidates_count}
                        </span>
                      </div>
                      <div className="flex justify-between text-[#9B9E98]">
                        <span>CONF / ENTROPY</span>
                        <span className="text-[#EAE9E4]">
                          {r.mean_confidence?.toFixed(3) ?? "—"} / {r.mean_entropy?.toFixed(3) ?? "—"}
                        </span>
                      </div>
                    </div>

                    <div>
                      <div className="h-1.5 w-full flex bg-[#121414] overflow-hidden rounded-[1px]">
                        {TERRAIN_CLASSES.map((c) => (
                          <div
                            key={c.key}
                            style={{
                              width: `${r.terrain_summary?.[c.key] ?? 0}%`,
                              backgroundColor: c.hex,
                            }}
                            title={`${c.label}: ${(r.terrain_summary?.[c.key] ?? 0).toFixed(1)}%`}
                          />
                        ))}
                      </div>
                      <div className="text-[10px] font-mono-tech text-[#696E68] mt-1">
                        {formatUtcTimestamp(r.generated_at_utc)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="px-3.5 py-2 bg-[#121414] border-t border-[#303431] flex items-center justify-between gap-2 text-[11px] font-mono-tech">
                  <Link
                    to={`/analyze?report=${encodeURIComponent(r.report_id)}`}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-[#202321] hover:bg-[#303431] border border-[#414642] text-[#EAE9E4] rounded-[2px]"
                  >
                    <Icon name="eye" className="text-[10px] text-[#C87951]" />
                    <span>Open Report</span>
                  </Link>

                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() => void openJsonInspector(r.report_id)}
                      className="inline-flex items-center gap-1 text-[#9B9E98] hover:text-[#EAE9E4]"
                    >
                      <Icon name="code" className="text-[10px]" />
                      <span>Inspect JSON</span>
                    </button>
                    <a
                      href={getDownloadReportUrl(r.report_id)}
                      className="inline-flex items-center gap-1 text-[#9B9E98] hover:text-[#EAE9E4]"
                    >
                      <Icon name="download" className="text-[10px]" />
                      <span>Download JSON</span>
                    </a>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* Native accessible modal dialog */}
      <dialog ref={dialogRef} aria-labelledby="json-modal-title" {...({ closedby: "any" } as Record<string, string>)}>
        {jsonModalDetail && (
          <div className="flex flex-col max-h-[88vh]">
            <div className="px-4 py-3 border-b border-[#303431] bg-[#121414] flex items-center justify-between gap-3">
              <h2 id="json-modal-title" className="text-[13px] font-mono-tech font-semibold text-[#EAE9E4]">
                Raw Provenance JSON — {jsonModalDetail.report_id}
              </h2>
              <form method="dialog">
                <button
                  type="submit"
                  className="px-2.5 py-1 border border-[#303431] text-[12px] text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px]"
                >
                  Close
                </button>
              </form>
            </div>
            <pre className="p-4 overflow-auto text-[11px] font-mono-tech text-[#EAE9E4] bg-[#0D0E0E] leading-relaxed">
              {JSON.stringify(jsonModalDetail.report, null, 2)}
            </pre>
          </div>
        )}
      </dialog>
    </div>
  );
};
