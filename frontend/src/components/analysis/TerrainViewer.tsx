import React, { useEffect, useRef, useState } from "react";
import { resolveArtifactUrl } from "../../services/api";
import type { ArtifactUrls, ObstacleCandidate, RegionItem } from "../../types/api";
import { IGNORE_CLASS_META, TERRAIN_CLASSES } from "../../utils/terrain";
import { Icon } from "../ui/Icon";

export type ViewerMode =
  | "original"
  | "mask"
  | "overlay"
  | "confidence"
  | "uncertainty"
  | "regions"
  | "candidates"
  | "dashboard"
  | "compare";

const VIEW_TABS: Array<{ id: ViewerMode; label: string; icon: string }> = [
  { id: "overlay", label: "Overlay", icon: "layer-group" },
  { id: "original", label: "Original", icon: "image" },
  { id: "mask", label: "Segmentation", icon: "border-all" },
  { id: "compare", label: "Compare Split", icon: "columns" },
  { id: "confidence", label: "Confidence", icon: "gauge-high" },
  { id: "uncertainty", label: "Uncertainty", icon: "wave-square" },
  { id: "regions", label: "Regions", icon: "draw-polygon" },
  { id: "candidates", label: "Candidates", icon: "vector-square" },
  { id: "dashboard", label: "9-Panel", icon: "table-cells" },
];

export interface TerrainViewerProps {
  imageId: string;
  imageWidth: number;
  imageHeight: number;
  artifactUrls: ArtifactUrls;
  selectedRegion: RegionItem | null;
  selectedCandidate: ObstacleCandidate | null;
  onClearSelection?: () => void;
}

export const TerrainViewer: React.FC<TerrainViewerProps> = ({
  imageId,
  imageWidth,
  imageHeight,
  artifactUrls,
  selectedRegion,
  selectedCandidate,
  onClearSelection,
}) => {
  const [mode, setMode] = useState<ViewerMode>("overlay");
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragOrigin = useRef<{ x: number; y: number; px: number; py: number }>({ x: 0, y: 0, px: 0, py: 0 });

  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.72);
  const [splitPct, setSplitPct] = useState<number>(50);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const onFs = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const fitToScreen = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFullscreen = async () => {
    const el = stageRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      await el.requestFullscreen?.();
    } else {
      await document.exitFullscreen?.();
    }
  };

  const getPrimaryUrl = (): string | null => {
    switch (mode) {
      case "original":
        return resolveArtifactUrl(artifactUrls.original_image);
      case "mask":
        return resolveArtifactUrl(artifactUrls.segmentation_mask);
      case "overlay":
        return resolveArtifactUrl(artifactUrls.terrain_overlay);
      case "confidence":
        return resolveArtifactUrl(artifactUrls.confidence_heatmap);
      case "uncertainty":
        return resolveArtifactUrl(artifactUrls.uncertainty_heatmap);
      case "regions":
        return resolveArtifactUrl(artifactUrls.region_boundaries);
      case "candidates":
        return resolveArtifactUrl(artifactUrls.big_rock_candidates);
      case "dashboard":
        return resolveArtifactUrl(artifactUrls.intelligence_summary_panel);
      default:
        return resolveArtifactUrl(artifactUrls.terrain_overlay);
    }
  };

  const originalUrl = resolveArtifactUrl(artifactUrls.original_image);
  const maskUrl = resolveArtifactUrl(artifactUrls.segmentation_mask);
  const overlayUrl = resolveArtifactUrl(artifactUrls.terrain_overlay);
  const activeUrl = getPrimaryUrl();

  // Only show spatial bbox SVG highlight over single-frame 512x512 views (not 9-panel)
  const supportsCoordOverlay = mode !== "dashboard";
  const highlightBox = selectedCandidate
    ? {
        id: selectedCandidate.candidate_id,
        bbox: selectedCandidate.bbox,
        centroid: selectedCandidate.centroid,
        label: `${selectedCandidate.candidate_id} (2D Candidate)`,
        color: "#E41A1C",
      }
    : selectedRegion
      ? {
          id: selectedRegion.region_id,
          bbox: selectedRegion.bbox,
          centroid: selectedRegion.centroid,
          label: `${selectedRegion.region_id} · ${selectedRegion.terrain_class}`,
          color: "#C87951",
        }
      : null;

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (zoom <= 1) return;
    setDragging(true);
    dragOrigin.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    setPan({
      x: dragOrigin.current.px + (e.clientX - dragOrigin.current.x),
      y: dragOrigin.current.py + (e.clientY - dragOrigin.current.y),
    });
  };

  const handlePointerUp = () => setDragging(false);

  return (
    <div ref={stageRef} className="mission-panel flex flex-col overflow-hidden bg-[#121414]">
      {/* Mode selector strip */}
      <div className="border-b border-[#303431] px-3 py-2 flex flex-wrap items-center justify-between gap-2 bg-[#181A19]">
        <div className="flex flex-wrap items-center gap-1">
          {VIEW_TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => {
                setMode(t.id);
                fitToScreen();
              }}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded-[2px] border transition-colors ${
                mode === t.id
                  ? "bg-[#202321] border-[#C87951] text-[#EAE9E4]"
                  : "border-transparent text-[#9B9E98] hover:text-[#EAE9E4] hover:bg-[#202321]/60"
              }`}
            >
              <Icon name={t.icon} className="text-[11px]" />
              <span>{t.label}</span>
            </button>
          ))}
        </div>

        {/* Zoom / Fit / Fullscreen controls */}
        <div className="flex items-center gap-1.5 font-mono-tech text-[11px]">
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))))}
            className="px-2 py-1 bg-[#121414] border border-[#303431] hover:border-[#414642] text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px]"
            aria-label="Zoom out"
          >
            <Icon name="magnifying-glass-minus" />
          </button>
          <span className="px-1.5 text-[#9B9E98] min-w-[44px] text-center">{Math.round(zoom * 100)}%</span>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(3, Number((z + 0.25).toFixed(2))))}
            className="px-2 py-1 bg-[#121414] border border-[#303431] hover:border-[#414642] text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px]"
            aria-label="Zoom in"
          >
            <Icon name="magnifying-glass-plus" />
          </button>
          <button
            type="button"
            onClick={fitToScreen}
            className="px-2 py-1 bg-[#121414] border border-[#303431] hover:border-[#414642] text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px]"
            title="Fit view"
          >
            Fit
          </button>
          <button
            type="button"
            onClick={() => void toggleFullscreen()}
            className="px-2 py-1 bg-[#121414] border border-[#303431] hover:border-[#414642] text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px]"
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            <Icon name={isFullscreen ? "compress" : "expand"} />
          </button>
        </div>
      </div>

      {/* Mode-specific secondary controls */}
      {(mode === "overlay" || mode === "compare" || highlightBox) && (
        <div className="border-b border-[#303431] px-4 py-2 bg-[#121414] flex flex-wrap items-center justify-between gap-4 text-[11px] font-mono-tech text-[#9B9E98]">
          {mode === "overlay" && (
            <label className="inline-flex items-center gap-2.5">
              <span>OVERLAY OPACITY</span>
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(overlayOpacity * 100)}
                onChange={(e) => setOverlayOpacity(Number(e.target.value) / 100)}
                className="w-28"
              />
              <span className="text-[#EAE9E4] w-9">{Math.round(overlayOpacity * 100)}%</span>
            </label>
          )}
          {mode === "compare" && (
            <label className="inline-flex items-center gap-2.5">
              <span>SPLIT (ORIGINAL | PREDICTED OVERLAY)</span>
              <input
                type="range"
                min={5}
                max={95}
                value={splitPct}
                onChange={(e) => setSplitPct(Number(e.target.value))}
                className="w-36"
              />
              <span className="text-[#EAE9E4] w-9">{splitPct}%</span>
            </label>
          )}
          {highlightBox && (
            <div className="inline-flex items-center gap-2 text-[#EAE9E4]">
              <Icon name="bullseye" className="text-[#C87951]" />
              <span>Focused: {highlightBox.label}</span>
              {onClearSelection && (
                <button
                  type="button"
                  onClick={onClearSelection}
                  className="text-[#9B9E98] hover:text-[#EAE9E4] underline ml-1"
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Main canvas */}
      <div
        className={`relative flex items-center justify-center bg-[#0D0E0E] overflow-hidden select-none ${
          isFullscreen ? "flex-1 min-h-[70vh]" : "min-h-[380px] sm:min-h-[500px]"
        } ${zoom > 1 ? "cursor-grab active:cursor-grabbing" : ""}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        {mode === "compare" ? (
          originalUrl && overlayUrl ? (
            <div
              className="relative max-w-full max-h-[580px] aspect-square"
              style={{
                width: "min(92vw, 540px)",
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "center center",
              }}
            >
              <img
                src={originalUrl}
                alt={`${imageId} original`}
                className="w-full h-full object-contain block"
                draggable={false}
              />
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ clipPath: `inset(0 0 0 ${splitPct}%)` }}
              >
                <img
                  src={overlayUrl}
                  alt={`${imageId} predicted overlay`}
                  className="w-full h-full object-contain block"
                  draggable={false}
                />
              </div>
              <div
                className="absolute top-0 bottom-0 w-px bg-[#C87951] pointer-events-none"
                style={{ left: `${splitPct}%` }}
              />
              <span className="absolute top-2 left-2 px-1.5 py-0.5 bg-[#0D0E0E]/85 border border-[#303431] text-[10px] font-mono-tech text-[#EAE9E4]">
                ORIGINAL
              </span>
              <span className="absolute top-2 right-2 px-1.5 py-0.5 bg-[#0D0E0E]/85 border border-[#303431] text-[10px] font-mono-tech text-[#EAE9E4]">
                PREDICTED OVERLAY
              </span>
            </div>
          ) : (
            <div className="p-8 text-center text-[13px] text-[#9B9E98]">
              Comparison unavailable: original or overlay artifact missing.
            </div>
          )
        ) : activeUrl ? (
          <div
            className={`relative flex items-center justify-center ${
              mode === "dashboard" ? "w-full max-w-[960px] p-2" : "w-[min(92vw,540px)] aspect-square"
            }`}
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "center center",
            }}
          >
            {mode === "overlay" && originalUrl && maskUrl ? (
              <>
                <img
                  src={originalUrl}
                  alt={`${imageId} base`}
                  className="w-full h-full object-contain block"
                  draggable={false}
                />
                <img
                  src={maskUrl}
                  alt={`${imageId} segmentation overlay`}
                  className="absolute inset-0 w-full h-full object-contain block pointer-events-none"
                  style={{ opacity: overlayOpacity }}
                  draggable={false}
                />
              </>
            ) : (
              <img
                src={activeUrl}
                alt={`${imageId} ${mode}`}
                className="w-full h-full object-contain block"
                draggable={false}
              />
            )}

            {/* SVG interactive Region / Obstacle Candidate overlay */}
            {supportsCoordOverlay && highlightBox && (
              <svg
                viewBox={`0 0 ${imageWidth || 512} ${imageHeight || 512}`}
                className="absolute inset-0 w-full h-full pointer-events-none"
              >
                <rect
                  x={highlightBox.bbox[0]}
                  y={highlightBox.bbox[1]}
                  width={highlightBox.bbox[2]}
                  height={highlightBox.bbox[3]}
                  fill="none"
                  stroke={highlightBox.color}
                  strokeWidth={2.5}
                />
                <circle
                  cx={highlightBox.centroid[0]}
                  cy={highlightBox.centroid[1]}
                  r={4.5}
                  fill="#EAE9E4"
                  stroke={highlightBox.color}
                  strokeWidth={1.5}
                />
              </svg>
            )}
          </div>
        ) : (
          <div className="p-8 text-center text-[13px] text-[#9B9E98]">
            <Icon name="circle-exclamation" className="mr-2 text-[#D0A064]" />
            Artifact for view &ldquo;{mode}&rdquo; is unavailable on disk.
          </div>
        )}
      </div>

      {/* Step 2 verified palette legend */}
      <div className="border-t border-[#303431] px-4 py-2.5 bg-[#181A19] flex flex-wrap items-center justify-between gap-3 text-[11px]">
        <div className="flex flex-wrap items-center gap-4">
          {TERRAIN_CLASSES.map((c) => (
            <span key={c.key} className="inline-flex items-center gap-1.5 text-[#EAE9E4]">
              <span className="w-3 h-3 rounded-[1px] border border-white/15" style={{ backgroundColor: c.hex }} />
              <span>{c.label}</span>
              <span className="text-[#696E68] font-mono-tech">({c.id})</span>
            </span>
          ))}
          <span className="inline-flex items-center gap-1.5 text-[#9B9E98]">
            <span
              className="w-3 h-3 rounded-[1px] border border-white/15"
              style={{ backgroundColor: IGNORE_CLASS_META.hex }}
            />
            <span>Ignore (255)</span>
          </span>
        </div>
        <div className="font-mono-tech text-[#696E68]">
          {imageWidth}×{imageHeight} px
        </div>
      </div>
    </div>
  );
};
