import React, { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { IntelligencePanel } from "../components/analysis/IntelligencePanel";
import { TerrainViewer } from "../components/analysis/TerrainViewer";
import { FadeContent } from "../components/motion/FadeContent";
import { Icon } from "../components/ui/Icon";
import { analyzeRoverImage, fetchReportDetail, fetchReports, getDownloadReportUrl } from "../services/api";
import type {
  ObstacleCandidate,
  RegionItem,
  ReportDetailResponse,
  ReportSummary,
  SystemStatusResponse,
} from "../types/api";

const ALLOWED_MIME_OR_EXT = /\.(png|jpg|jpeg|webp|bmp|tif|tiff)$/i;

const MODEL_CHOICES = [
  { id: "deeplabv3plus", label: "DeepLabV3+ (Baseline)", note: "ResNet50 · Default" },
  { id: "unet", label: "U-Net (Baseline)", note: "ResNet34 · Fast" },
  { id: "deeplabv3plus_optimized", label: "DeepLabV3+ Optimized", note: "CE+Dice MSL" },
  { id: "unet_optimized", label: "U-Net Optimized", note: "CE+Dice Balanced" },
];

export interface TerrainAnalysisPageProps {
  systemStatus: SystemStatusResponse | null;
  activeModel: string;
  onModelChange: (modelId: string) => void;
  onAnalysisComplete?: () => void;
}

export const TerrainAnalysisPage: React.FC<TerrainAnalysisPageProps> = ({
  systemStatus,
  activeModel,
  onModelChange,
  onAnalysisComplete,
}) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const reportParam = searchParams.get("report");

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const [analyzing, setAnalyzing] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [detail, setDetail] = useState<ReportDetailResponse | null>(null);
  const [recentOptions, setRecentOptions] = useState<ReportSummary[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<RegionItem | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<ObstacleCandidate | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load recent existing reports list and initial detail if ?report= is set or load latest
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetchReports();
        if (!active) return;
        setRecentOptions(res.reports);
        const targetId = reportParam ?? res.reports[0]?.report_id;
        if (targetId) {
          setLoadingReport(true);
          const rep = await fetchReportDetail(targetId);
          if (active) {
            setDetail(rep);
            setSelectedRegion(null);
            setSelectedCandidate(null);
          }
        }
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : "Failed to load report.");
      } finally {
        if (active) setLoadingReport(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [reportParam]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const acceptFile = (f: File) => {
    if (!ALLOWED_MIME_OR_EXT.test(f.name) && !f.type.startsWith("image/")) {
      setError(`Invalid file "${f.name}". Please upload PNG, JPG, JPEG, WEBP, BMP, or TIFF Mars imagery.`);
      return;
    }
    if (f.size === 0) {
      setError("Uploaded file is empty (0 bytes).");
      return;
    }
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const clearSelectedFile = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile || analyzing) return;
    setAnalyzing(true);
    setError(null);
    try {
      const result = await analyzeRoverImage(selectedFile, activeModel);
      setDetail(result);
      setSelectedRegion(null);
      setSelectedCandidate(null);
      setSearchParams({ report: result.report_id });
      const refreshed = await fetchReports();
      setRecentOptions(refreshed.reports);
      onAnalysisComplete?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference failed on backend.");
    } finally {
      setAnalyzing(false);
    }
  };

  const deviceSummary = systemStatus
    ? systemStatus.cuda_available && systemStatus.gpu_name
      ? `${systemStatus.gpu_name} (${systemStatus.gpu_vram_gb ?? "?"} GB VRAM)`
      : `CPU (${systemStatus.active_device})`
    : "Detecting...";

  return (
    <div className="space-y-5">
      {/* Control Bar: Understated Dropzone + Model Selector + Saved Report Switcher */}
      <section className="mission-panel p-4 space-y-4">
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-stretch">
          {/* Understated Dropzone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              const f = e.dataTransfer.files?.[0];
              if (f) acceptFile(f);
            }}
            className={`xl:col-span-5 border border-dashed rounded-[2px] p-3.5 flex items-center justify-between gap-3 transition-colors ${
              dragActive ? "border-[#C87951] bg-[#C87951]/10" : "border-[#414642] bg-[#121414]"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,image/*"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) acceptFile(f);
              }}
              className="hidden"
            />

            {selectedFile && previewUrl ? (
              <div className="flex items-center gap-3 min-w-0">
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className="w-14 h-14 object-cover border border-[#303431] shrink-0"
                />
                <div className="min-w-0">
                  <div className="text-[12px] font-mono-tech text-[#EAE9E4] truncate">{selectedFile.name}</div>
                  <div className="text-[11px] font-mono-tech text-[#9B9E98]">
                    {(selectedFile.size / 1024).toFixed(1)} KB · Ready for inference
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 bg-[#181A19] border border-[#303431] flex items-center justify-center shrink-0 text-[#9B9E98]">
                  <Icon name="upload" />
                </div>
                <div className="min-w-0">
                  <div className="text-[13px] font-medium text-[#EAE9E4]">Upload Mars Rover Frame</div>
                  <div className="text-[11px] text-[#9B9E98]">
                    Drag &amp; drop PNG/JPG or browse local filesystem
                  </div>
                </div>
              </div>
            )}

            <div className="flex items-center gap-2 shrink-0">
              {selectedFile && (
                <button
                  type="button"
                  onClick={clearSelectedFile}
                  disabled={analyzing}
                  className="px-2.5 py-1.5 border border-[#303431] hover:border-[#C76E64] text-[11px] font-mono-tech text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px] disabled:opacity-50"
                >
                  Clear
                </button>
              )}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={analyzing}
                className="px-3 py-1.5 bg-[#202321] hover:bg-[#303431] border border-[#414642] text-[12px] text-[#EAE9E4] rounded-[2px] disabled:opacity-50"
              >
                Browse File
              </button>
            </div>
          </div>

          {/* Model Selection & Device Readout */}
          <div className="xl:col-span-5 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1.5">
                Segmentation Architecture
              </label>
              <select
                value={activeModel}
                onChange={(e) => onModelChange(e.target.value)}
                disabled={analyzing}
                className="w-full pl-3 pr-8 py-2 bg-[#121414] border border-[#303431] text-[12px] text-[#EAE9E4] rounded-[2px] focus:border-[#C87951] focus:outline-none truncate"
              >
                {MODEL_CHOICES.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label} — {m.note}
                  </option>
                ))}
              </select>
              <div className="text-[10px] font-mono-tech text-[#696E68] mt-1 truncate">
                Device: {deviceSummary}
              </div>
            </div>

            <div>
              <label className="block text-[10px] font-mono-tech uppercase text-[#9B9E98] mb-1.5">
                Inspect Existing Report
              </label>
              <select
                value={detail?.report_id ?? ""}
                onChange={(e) => {
                  if (e.target.value) setSearchParams({ report: e.target.value });
                }}
                className="w-full pl-3 pr-8 py-2 bg-[#121414] border border-[#303431] text-[12px] font-mono-tech text-[#EAE9E4] rounded-[2px] focus:border-[#C87951] focus:outline-none truncate"
              >
                {recentOptions.length === 0 && <option value="">No reports found</option>}
                {recentOptions.map((opt) => (
                  <option key={opt.report_id} value={opt.report_id}>
                    {opt.report_id} ({opt.mission ?? "Upload"})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Execute Button & Export */}
          <div className="xl:col-span-2 flex flex-col justify-between gap-2">
            <button
              type="button"
              onClick={() => void handleRunAnalysis()}
              disabled={!selectedFile || analyzing}
              className="w-full h-10 inline-flex items-center justify-center gap-2 px-4 bg-[#C87951] hover:bg-[#A65E40] disabled:bg-[#202321] disabled:text-[#696E68] text-[#0D0E0E] font-semibold text-[13px] rounded-[2px] transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <Icon name={analyzing ? "spinner" : "play"} spin={analyzing} className="text-[12px]" />
              <span>{analyzing ? "Analyzing Terrain..." : "Run Analysis"}</span>
            </button>

            {detail && (
              <a
                href={getDownloadReportUrl(detail.report_id)}
                className="w-full py-1.5 inline-flex items-center justify-center gap-2 bg-[#121414] hover:bg-[#202321] border border-[#303431] text-[11px] font-mono-tech text-[#9B9E98] hover:text-[#EAE9E4] rounded-[2px]"
              >
                <Icon name="download" className="text-[11px]" />
                <span>Download JSON</span>
              </a>
            )}
          </div>
        </div>

        {/* Real processing / error feedback */}
        {analyzing && (
          <div className="px-3.5 py-2.5 bg-[#121414] border border-[#C87951]/50 text-[12px] font-mono-tech text-[#EAE9E4] flex items-center gap-2.5">
            <Icon name="spinner" spin className="text-[#C87951]" />
            <span>
              Executing live PyTorch inference ({activeModel}), connected-component filtering, entropy computation,
              and 9-artifact rendering on backend...
            </span>
          </div>
        )}

        {error && (
          <div className="px-3.5 py-2.5 bg-[#C76E64]/10 border border-[#C76E64]/50 text-[12px] text-[#EAE9E4] flex items-center gap-2">
            <Icon name="circle-exclamation" className="text-[#C76E64]" />
            <span>{error}</span>
          </div>
        )}
      </section>

      {/* Main Split Workspace: Interactive Canvas + Scientific Report Panel */}
      {loadingReport && !detail ? (
        <div className="mission-panel p-12 text-center text-[#9B9E98] font-mono-tech text-[13px]">
          <Icon name="spinner" spin className="mr-2 text-[#C87951]" />
          Loading terrain intelligence report...
        </div>
      ) : detail ? (
        <FadeContent>
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
            <div className="xl:col-span-7 xl:sticky xl:top-20">
              <TerrainViewer
                imageId={detail.report.image_metadata.image_id}
                imageWidth={detail.report.image_metadata.width}
                imageHeight={detail.report.image_metadata.height}
                artifactUrls={detail.artifact_urls}
                selectedRegion={selectedRegion}
                selectedCandidate={selectedCandidate}
                onClearSelection={() => {
                  setSelectedRegion(null);
                  setSelectedCandidate(null);
                }}
              />
            </div>

            <div className="xl:col-span-5">
              <IntelligencePanel
                report={detail.report}
                mission={detail.mission}
                selectedRegionId={selectedRegion?.region_id ?? null}
                selectedCandidateId={selectedCandidate?.candidate_id ?? null}
                onSelectRegion={(reg) => {
                  setSelectedRegion(reg);
                  setSelectedCandidate(null);
                }}
                onSelectCandidate={(cand) => {
                  setSelectedCandidate(cand);
                  setSelectedRegion(null);
                }}
              />
            </div>
          </div>
        </FadeContent>
      ) : (
        <div className="mission-panel p-12 text-center text-[#9B9E98] text-[13px]">
          Upload a Mars rover image above or select a stored analysis to inspect predictions.
        </div>
      )}
    </div>
  );
};
