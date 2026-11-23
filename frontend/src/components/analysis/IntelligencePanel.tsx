import React from "react";
import type { ObstacleCandidate, RegionItem, TerrainIntelligenceReport, TerrainKey } from "../../types/api";
import { TERRAIN_CLASSES, formatConcernLabel } from "../../utils/terrain";
import { Icon } from "../ui/Icon";

export interface IntelligencePanelProps {
  report: TerrainIntelligenceReport;
  mission: string | null;
  selectedRegionId: string | null;
  selectedCandidateId: string | null;
  onSelectRegion: (region: RegionItem | null) => void;
  onSelectCandidate: (candidate: ObstacleCandidate | null) => void;
}

export const IntelligencePanel: React.FC<IntelligencePanelProps> = ({
  report,
  mission,
  selectedRegionId,
  selectedCandidateId,
  onSelectRegion,
  onSelectCandidate,
}) => {
  const { analysis, image_metadata } = report;
  const ts = analysis.terrain_summary;
  const td = analysis.terrain_details;
  const conf = analysis.confidence;
  const unc = analysis.uncertainty;
  const trav = analysis.traversability;
  const concernStyle = formatConcernLabel(trav.assessment);

  const histCounts = conf.confidence_histogram.pixel_counts;
  const maxHist = Math.max(...histCounts, 1);

  return (
    <div className="space-y-4">
      {/* 1. Experimental Visual Concern Assessment */}
      <section className={`border rounded-[3px] p-4 ${concernStyle.borderClass}`}>
        <div className="flex flex-wrap items-start justify-between gap-2 mb-2">
          <div>
            <div className="text-[10px] font-mono-tech uppercase tracking-[0.14em] text-[#9B9E98]">
              Experimental 2D Visual Concern Assessment
            </div>
            <div className={`text-[18px] font-semibold mt-0.5 ${concernStyle.toneClass}`}>
              {concernStyle.label}
            </div>
          </div>
          <div className="text-right font-mono-tech">
            <div className="text-[10px] text-[#9B9E98]">CONCERN INDEX</div>
            <div className="text-[15px] text-[#EAE9E4] font-semibold">
              {trav.experimental_visual_concern_index.toFixed(3)}
            </div>
          </div>
        </div>

        <div className="mt-2 py-2 px-3 bg-[#0D0E0E]/80 border border-[#C76E64]/40 rounded-[2px] flex items-center justify-between gap-2 text-[12px]">
          <span className="font-mono-tech font-semibold text-[#C76E64]">
            <Icon name="shield-halved" className="mr-2" />
            Physical safety verified: {trav.physical_safety_verified ? "YES" : "NO"}
          </span>
          <span className="text-[11px] font-mono-tech text-[#9B9E98]">2D Image Only</span>
        </div>

        <p className="text-[12px] text-[#9B9E98] mt-2.5 leading-relaxed">
          {trav.scientific_disclaimer}
        </p>

        <div className="mt-3 pt-2.5 border-t border-[#303431]/80 grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px] font-mono-tech items-end">
          <div>
            <div className="text-[#696E68] truncate">Terrain Comp.</div>
            <div className="text-[#EAE9E4] mt-0.5">{trav.component_scores.terrain_composition_score.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-[#696E68] truncate">Rock Density</div>
            <div className="text-[#EAE9E4] mt-0.5">{trav.component_scores.big_rock_candidate_density_score.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-[#696E68] truncate">Sand Conc.</div>
            <div className="text-[#EAE9E4] mt-0.5">{trav.component_scores.sand_concentration_score.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-[#696E68] truncate">Uncertainty</div>
            <div className="text-[#EAE9E4] mt-0.5">{trav.component_scores.segmentation_uncertainty_score.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-[#696E68] truncate">Regions</div>
            <div className="text-[#EAE9E4] mt-0.5">{trav.component_scores.region_fragmentation_score.toFixed(3)}</div>
          </div>
        </div>

        <div className="mt-2.5 text-[11px] text-[#9B9E98]">
          <span className="font-mono-tech uppercase text-[#696E68] mr-1.5">Unavailable measurements:</span>
          {trav.missing_measurements.join(" · ")}
        </div>
      </section>

      {/* 2. Terrain Composition */}
      <section className="mission-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-3 border-b border-[#303431]">
          <h3 className="text-[13px] font-semibold tracking-wide text-[#EAE9E4] uppercase flex items-center gap-2">
            <Icon name="chart-pie" className="text-[#C87951] text-[12px]" />
            <span>Terrain Composition</span>
          </h3>
          <div className="font-mono-tech text-[11px] text-[#9B9E98]">
            Valid: {td.valid_pixel_percentage.toFixed(1)}% ({td.total_valid_pixels.toLocaleString()} px)
          </div>
        </div>

        <div className="space-y-2.5">
          {TERRAIN_CLASSES.map((c) => {
            const pct = ts[c.key as TerrainKey] ?? 0;
            const pxCount = td.class_pixel_counts[c.key as TerrainKey] ?? 0;
            return (
              <div key={c.key} className="space-y-1">
                <div className="flex items-center justify-between text-[12px]">
                  <span className="inline-flex items-center gap-2 text-[#EAE9E4]">
                    <span className="w-2.5 h-2.5 rounded-[1px]" style={{ backgroundColor: c.hex }} />
                    <span>{c.label}</span>
                  </span>
                  <span className="font-mono-tech text-[#EAE9E4]">
                    {pct.toFixed(2)}%{" "}
                    <span className="text-[#696E68]">({pxCount.toLocaleString()} px)</span>
                  </span>
                </div>
                <div className="h-1.5 w-full bg-[#121414] overflow-hidden rounded-[1px]">
                  <div
                    className="h-full transition-all duration-300"
                    style={{ width: `${Math.min(100, pct)}%`, backgroundColor: c.hex }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 pt-3 border-t border-[#303431] grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
          <div>
            <div className="text-[#696E68] font-mono-tech uppercase">Dominant</div>
            <div className="text-[#EAE9E4] font-medium capitalize mt-0.5">
              {td.dominant_terrain?.replace("_", " ") ?? "None"}
            </div>
          </div>
          <div>
            <div className="text-[#696E68] font-mono-tech uppercase">Secondary</div>
            <div className="text-[#EAE9E4] font-medium capitalize mt-0.5">
              {td.secondary_terrain?.replace("_", " ") ?? "None"}
            </div>
          </div>
          <div>
            <div className="text-[#696E68] font-mono-tech uppercase">Diversity (H)</div>
            <div className="text-[#EAE9E4] font-mono-tech mt-0.5">
              {td.terrain_diversity_shannon_nats.toFixed(3)} ({td.terrain_diversity_normalized.toFixed(2)})
            </div>
          </div>
          <div>
            <div className="text-[#696E68] font-mono-tech uppercase">Boundary Density</div>
            <div className="text-[#EAE9E4] font-mono-tech mt-0.5">{td.boundary_density_pct.toFixed(2)}%</div>
          </div>
        </div>
      </section>

      {/* 3. 2D Big Rock Obstacle Candidates */}
      <section className="mission-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-2.5 border-b border-[#303431]">
          <h3 className="text-[13px] font-semibold tracking-wide text-[#EAE9E4] uppercase flex items-center gap-2">
            <Icon name="cubes" className="text-[#E41A1C] text-[12px]" />
            <span>Big Rock Candidates (2D Visual Candidates Only)</span>
          </h3>
          <span className="font-mono-tech text-[11px] text-[#9B9E98]">
            {analysis.obstacle_candidates.length} detected
          </span>
        </div>

        {analysis.obstacle_candidates.length === 0 ? (
          <div className="py-3 text-[12px] text-[#9B9E98]">
            No Big Rock visual candidates exceeded configured area/confidence thresholds.
          </div>
        ) : (
          <div className="table-wrapper" tabIndex={0} role="region" aria-label="2D Big Rock Candidates Table">
            <table className="w-full text-left text-[12px] border-collapse">
              <thead>
                <tr className="border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                  <th className="py-2 pr-3">Candidate</th>
                  <th className="py-2 pr-3">Area</th>
                  <th className="py-2 pr-3">Conf</th>
                  <th className="py-2 pr-3">BBox [x,y,w,h]</th>
                  <th className="py-2 pr-3">Shape (Circ / Sol)</th>
                  <th className="py-2">Position</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#303431]/60 font-mono-tech text-[11px]">
                {analysis.obstacle_candidates.map((cand) => {
                  const active = selectedCandidateId === cand.candidate_id;
                  return (
                    <tr
                      key={cand.candidate_id}
                      onClick={() => onSelectCandidate(active ? null : cand)}
                      className={`cursor-pointer transition-colors ${
                        active ? "bg-[#E41A1C]/15 text-[#EAE9E4]" : "hover:bg-[#202321]"
                      }`}
                    >
                      <td className="py-2 pr-3 font-semibold text-[#EAE9E4]">{cand.candidate_id}</td>
                      <td className="py-2 pr-3">{cand.pixel_area} px</td>
                      <td className="py-2 pr-3">{cand.confidence.toFixed(3)}</td>
                      <td className="py-2 pr-3">[{cand.bbox.join(", ")}]</td>
                      <td className="py-2 pr-3">
                        {cand.shape.circularity.toFixed(2)} / {cand.shape.solidity.toFixed(2)}
                      </td>
                      <td className="py-2 text-[#9B9E98]">{cand.relative_position}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 4. Confidence & Predictive Uncertainty */}
      <section className="mission-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-3 border-b border-[#303431]">
          <h3 className="text-[13px] font-semibold tracking-wide text-[#EAE9E4] uppercase flex items-center gap-2">
            <Icon name="wave-square" className="text-[#C87951] text-[12px]" />
            <span>Confidence &amp; Predictive Uncertainty</span>
          </h3>
          <span className="font-mono-tech text-[11px] text-[#D0A064]">Uncalibrated Softmax</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px] font-mono-tech mb-3">
          <div className="p-2 bg-[#121414] border border-[#303431] rounded-[2px]">
            <div className="text-[#696E68]">MEAN CONF</div>
            <div className="text-[14px] text-[#EAE9E4] mt-0.5">{conf.mean_confidence.toFixed(4)}</div>
          </div>
          <div className="p-2 bg-[#121414] border border-[#303431] rounded-[2px]">
            <div className="text-[#696E68]">LOW CONF (&lt;{conf.thresholds.medium_min})</div>
            <div className="text-[14px] text-[#EAE9E4] mt-0.5">{conf.low_confidence_pixel_percentage.toFixed(2)}%</div>
          </div>
          <div className="p-2 bg-[#121414] border border-[#303431] rounded-[2px]">
            <div className="text-[#696E68]">MEAN ENTROPY</div>
            <div className="text-[14px] text-[#EAE9E4] mt-0.5">{unc.mean_entropy.toFixed(4)} nats</div>
          </div>
          <div className="p-2 bg-[#121414] border border-[#303431] rounded-[2px]">
            <div className="text-[#696E68]">NORM ENTROPY</div>
            <div className="text-[14px] text-[#EAE9E4] mt-0.5">{unc.mean_normalized_entropy.toFixed(4)}</div>
          </div>
        </div>

        {/* Confidence tiers bar */}
        <div className="text-[11px] font-mono-tech text-[#9B9E98] mb-1.5 flex justify-between">
          <span>High (&ge;{conf.thresholds.high_min}): {conf.confidence_categories_pct.high.toFixed(1)}%</span>
          <span>Med: {conf.confidence_categories_pct.medium.toFixed(1)}%</span>
          <span>Low: {conf.confidence_categories_pct.low.toFixed(1)}%</span>
        </div>
        <div className="h-2 w-full flex overflow-hidden rounded-[1px] bg-[#121414] mb-4">
          <div style={{ width: `${conf.confidence_categories_pct.high}%` }} className="bg-[#829B79]" />
          <div style={{ width: `${conf.confidence_categories_pct.medium}%` }} className="bg-[#D0A064]" />
          <div style={{ width: `${conf.confidence_categories_pct.low}%` }} className="bg-[#C76E64]" />
        </div>

        {/* 10-bin Softmax Confidence Histogram */}
        <div className="text-[10px] font-mono-tech uppercase text-[#696E68] mb-1">
          Softmax Confidence Histogram (0.0 → 1.0)
        </div>
        <div className="flex items-end gap-1 h-16 px-1 pt-2 bg-[#121414] border border-[#303431] rounded-[2px]">
          {histCounts.map((cnt, i) => {
            const pctH = Math.round((cnt / maxHist) * 100);
            return (
              <div
                key={i}
                className="flex-1 bg-[#C87951]/80 hover:bg-[#C87951] transition-colors rounded-t-[1px]"
                style={{ height: `${Math.max(pctH, cnt > 0 ? 6 : 0)}%` }}
                title={`Bin ${(i * 0.1).toFixed(1)}–${((i + 1) * 0.1).toFixed(1)}: ${cnt.toLocaleString()} px`}
              />
            );
          })}
        </div>

        <p className="text-[11px] text-[#9B9E98] mt-2.5 leading-relaxed">
          <Icon name="circle-info" className="mr-1.5 text-[#9B9E98]" />
          {conf.calibration_note}
        </p>
      </section>

      {/* 5. Selectable Region Analysis Table */}
      <section className="mission-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-2.5 border-b border-[#303431]">
          <h3 className="text-[13px] font-semibold tracking-wide text-[#EAE9E4] uppercase flex items-center gap-2">
            <Icon name="draw-polygon" className="text-[#C87951] text-[12px]" />
            <span>Connected Regions (Click Row to Highlight)</span>
          </h3>
          <span className="font-mono-tech text-[11px] text-[#9B9E98]">
            Retained: {analysis.regions.length}
            {analysis.region_filtering
              ? ` · Filtered small: ${analysis.region_filtering.filtered_small_component_count}`
              : ""}
          </span>
        </div>

        {analysis.regions.length === 0 ? (
          <div className="py-3 text-[12px] text-[#9B9E98]">No contiguous terrain regions retained above minimum area.</div>
        ) : (
          <div className="table-wrapper max-h-[260px]" tabIndex={0} role="region" aria-label="Terrain Regions Table">
            <table className="w-full text-left text-[12px] border-collapse">
              <thead className="sticky top-0 bg-[#181A19] z-10">
                <tr className="border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                  <th className="py-2 pr-3">ID</th>
                  <th className="py-2 pr-3">Class</th>
                  <th className="py-2 pr-3">Area</th>
                  <th className="py-2 pr-3">Centroid</th>
                  <th className="py-2 pr-3">Conf</th>
                  <th className="py-2">Position</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#303431]/60 font-mono-tech text-[11px]">
                {analysis.regions.map((reg) => {
                  const active = selectedRegionId === reg.region_id;
                  return (
                    <tr
                      key={reg.region_id}
                      onClick={() => onSelectRegion(active ? null : reg)}
                      className={`cursor-pointer transition-colors ${
                        active ? "bg-[#C87951]/15 text-[#EAE9E4]" : "hover:bg-[#202321]"
                      }`}
                    >
                      <td className="py-1.5 pr-3 font-semibold text-[#EAE9E4]">{reg.region_id}</td>
                      <td className="py-1.5 pr-3 capitalize">{reg.terrain_class.replace("_", " ")}</td>
                      <td className="py-1.5 pr-3">
                        {reg.pixel_area.toLocaleString()} px ({reg.image_coverage_pct.toFixed(1)}%)
                      </td>
                      <td className="py-1.5 pr-3">
                        ({reg.centroid[0].toFixed(0)}, {reg.centroid[1].toFixed(0)})
                      </td>
                      <td className="py-1.5 pr-3">{reg.mean_confidence.toFixed(3)}</td>
                      <td className="py-1.5 text-[#9B9E98]">{reg.relative_position}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 6. Image & Model Provenance Footer */}
      <section className="mission-panel p-3.5 text-[11px] font-mono-tech text-[#9B9E98] flex flex-wrap gap-x-6 gap-y-1.5">
        <span>IMAGE: <strong className="text-[#EAE9E4]">{image_metadata.image_id}</strong></span>
        <span>MISSION: <strong className="text-[#EAE9E4]">{mission ?? "External / Uncatalogued"}</strong></span>
        <span>MODEL: <strong className="text-[#EAE9E4]">{report.model}</strong></span>
        <span>DEVICE: <strong className="text-[#EAE9E4]">{report.device}</strong></span>
        {image_metadata.inference_latency_ms !== null && (
          <span>LATENCY: <strong className="text-[#EAE9E4]">{image_metadata.inference_latency_ms.toFixed(1)} ms</strong></span>
        )}
      </section>
    </div>
  );
};
