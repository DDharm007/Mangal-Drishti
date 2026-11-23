import React, { useState } from "react";
import { TERRAIN_CLASSES } from "../../utils/terrain";

export interface ConfusionMatrixHeatmapProps {
  title: string;
  subtitle?: string;
  matrix: number[][];
  rowNormalized: number[][];
  labels?: string[];
}

function formatCompactPixels(val: number): string {
  if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`;
  if (val >= 1_000) return `${(val / 1_000).toFixed(1)}K`;
  return String(val);
}

function cellBackground(fraction: number, isDiag: boolean): string {
  const clamped = Math.max(0, Math.min(1, fraction));
  if (isDiag) {
    // Charcoal -> Muted Terracotta -> Primary Terracotta (#C87951)
    const alpha = 0.14 + clamped * 0.72;
    return `rgba(200, 121, 81, ${alpha.toFixed(3)})`;
  }
  // Off-diagonal confusion errors: muted warm amber/rust tint proportional to leakage
  if (clamped < 0.002) return "rgba(24, 26, 25, 0.9)";
  const alpha = 0.08 + clamped * 0.52;
  return `rgba(199, 110, 100, ${alpha.toFixed(3)})`;
}

export const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixHeatmapProps> = ({
  title,
  subtitle,
  matrix,
  rowNormalized,
  labels = ["Soil", "Bedrock", "Sand", "Big Rock"],
}) => {
  const [hovered, setHovered] = useState<{ r: number; c: number } | null>(null);

  const displayLabels = labels.map((l) =>
    l
      .split(" ")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  );

  const activeCell = hovered
    ? {
        trueLabel: displayLabels[hovered.r],
        predLabel: displayLabels[hovered.c],
        pixels: matrix[hovered.r]?.[hovered.c] ?? 0,
        pct: ((rowNormalized[hovered.r]?.[hovered.c] ?? 0) * 100).toFixed(2),
      }
    : null;

  return (
    <div className="border border-[#303431] bg-[#121414] rounded-[3px] p-3.5 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2 border-b border-[#303431] pb-2">
        <div>
          <div className="text-[13px] font-semibold text-[#EAE9E4]">{title}</div>
          {subtitle && <div className="text-[11px] font-mono-tech text-[#9B9E98]">{subtitle}</div>}
        </div>
        <div className="text-[10px] font-mono-tech text-[#696E68] uppercase">Rows: True · Cols: Pred</div>
      </div>

      <div className="table-wrapper" tabIndex={0} role="region" aria-label={title}>
        <table className="w-full border-collapse text-[11px] font-mono-tech min-w-[340px]">
          <thead>
            <tr>
              <th className="p-1.5 text-left text-[10px] text-[#696E68] uppercase font-normal">True \ Pred</th>
              {displayLabels.map((lbl, cIdx) => (
                <th key={lbl} className="p-1.5 text-center font-medium text-[#9B9E98]">
                  <span className="inline-flex items-center gap-1.5 justify-center">
                    <span
                      className="w-2 h-2 rounded-[1px] shrink-0"
                      style={{ backgroundColor: TERRAIN_CLASSES[cIdx]?.hex ?? "#696E68" }}
                    />
                    <span>{lbl}</span>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayLabels.map((rowLbl, rIdx) => (
              <tr key={rowLbl}>
                <th className="p-1.5 text-left font-medium text-[#EAE9E4] whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-[1px] shrink-0"
                      style={{ backgroundColor: TERRAIN_CLASSES[rIdx]?.hex ?? "#696E68" }}
                    />
                    <span>{rowLbl}</span>
                  </span>
                </th>
                {displayLabels.map((_, cIdx) => {
                  const frac = rowNormalized[rIdx]?.[cIdx] ?? 0;
                  const px = matrix[rIdx]?.[cIdx] ?? 0;
                  const isDiag = rIdx === cIdx;
                  const isActive = hovered?.r === rIdx && hovered?.c === cIdx;
                  return (
                    <td key={cIdx} className="p-0.5">
                      <button
                        type="button"
                        onMouseEnter={() => setHovered({ r: rIdx, c: cIdx })}
                        onFocus={() => setHovered({ r: rIdx, c: cIdx })}
                        onMouseLeave={() => setHovered(null)}
                        style={{ backgroundColor: cellBackground(frac, isDiag) }}
                        className={`w-full py-2 px-1 text-center rounded-[2px] border transition-colors ${
                          isActive
                            ? "border-[#EAE9E4]"
                            : isDiag
                              ? "border-[#C87951]/45"
                              : "border-[#303431]/70"
                        }`}
                      >
                        <div className="text-[11.5px] font-semibold text-[#EAE9E4] leading-tight">
                          {(frac * 100).toFixed(1)}%
                        </div>
                        <div className="text-[9.5px] text-[#EAE9E4]/70 leading-tight mt-0.5">
                          {formatCompactPixels(px)}
                        </div>
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Hover Telemetry Readout Bar */}
      <div className="px-2.5 py-1.5 bg-[#181A19] border border-[#303431] rounded-[2px] text-[11px] font-mono-tech text-[#9B9E98] min-h-[30px] flex items-center justify-between gap-2">
        {activeCell ? (
          <>
            <span>
              True <strong className="text-[#EAE9E4]">{activeCell.trueLabel}</strong> → Pred{" "}
              <strong className="text-[#EAE9E4]">{activeCell.predLabel}</strong>
            </span>
            <span className="text-[#EAE9E4]">
              {activeCell.pixels.toLocaleString()} px ({activeCell.pct}%)
            </span>
          </>
        ) : (
          <span className="text-[#696E68]">Hover or focus any matrix cell to inspect exact pixel count &amp; recall fraction.</span>
        )}
      </div>
    </div>
  );
};
