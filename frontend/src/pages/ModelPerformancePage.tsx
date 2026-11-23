import React, { useEffect, useState } from "react";
import { FadeContent } from "../components/motion/FadeContent";
import { Icon } from "../components/ui/Icon";
import { fetchModelsPerformance, resolveArtifactUrl } from "../services/api";
import type { ModelMetricsBlock, ModelsPerformanceResponse } from "../types/api";
import { TERRAIN_CLASSES } from "../utils/terrain";

type SplitTab = "test_combined" | "test_msl" | "test_mer" | "validation" | "benchmarks";

interface ComparisonRow {
  label: string;
  splitNote: string;
  checkpointType: "Baseline" | "Optimized";
  paramsM: string;
  metrics: ModelMetricsBlock;
}

export const ModelPerformancePage: React.FC = () => {
  const [data, setData] = useState<ModelsPerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<SplitTab>("test_combined");

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        setLoading(true);
        const res = await fetchModelsPerformance();
        if (active) setData(res);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : "Failed to load evaluation metrics.");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="mission-panel p-10 text-center font-mono-tech text-[13px] text-[#9B9E98]">
        Loading Step 3 &amp; Step 4 evaluation records...
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="p-4 border border-[#C76E64]/45 bg-[#C76E64]/10 text-[13px] text-[#EAE9E4]">
        {error ?? "Evaluation records unavailable."}
      </div>
    );
  }

  const dlOpt = data.optimization?.experiments?.deeplabv3plus_ce_dice_balanced;
  const unetOpt = data.optimization?.experiments?.unet_ce_dice_balanced;

  const unetParams = `${((data.benchmarks?.models?.unet?.parameter_count ?? 24521796) / 1e6).toFixed(2)}M`;
  const dlParams = `${((data.benchmarks?.models?.deeplabv3plus?.parameter_count ?? 40347812) / 1e6).toFixed(2)}M`;

  const getRowsForSplit = (tab: SplitTab): { title: string; subtitle: string; rows: ComparisonRow[] } => {
    switch (tab) {
      case "validation":
        return {
          title: "Validation Split Comparison (363 MSL Curiosity Images)",
          subtitle: "Used strictly for epoch checkpoint selection and controlled Step 4 train/val optimization.",
          rows: [
            ...(dlOpt
              ? [
                  {
                    label: "DeepLabV3+ + ResNet50 (Baseline Ep. 10)",
                    splitNote: "Val 363",
                    checkpointType: "Baseline" as const,
                    paramsM: dlParams,
                    metrics: dlOpt.baseline_val,
                  },
                  {
                    label: "DeepLabV3+ + ResNet50 (Optimized CE+Dice Ep. 4)",
                    splitNote: "Val 363",
                    checkpointType: "Optimized" as const,
                    paramsM: dlParams,
                    metrics: dlOpt.optimized_val,
                  },
                ]
              : []),
            ...(unetOpt
              ? [
                  {
                    label: "U-Net + ResNet34 (Baseline Ep. 6)",
                    splitNote: "Val 363",
                    checkpointType: "Baseline" as const,
                    paramsM: unetParams,
                    metrics: unetOpt.baseline_val,
                  },
                  {
                    label: "U-Net + ResNet34 (Optimized CE+Dice Ep. 2)",
                    splitNote: "Val 363",
                    checkpointType: "Optimized" as const,
                    paramsM: unetParams,
                    metrics: unetOpt.optimized_val,
                  },
                ]
              : []),
          ],
        };
      case "test_msl":
        return {
          title: "Expert Test — MSL (Curiosity) Subset (322 Images)",
          subtitle: "In-domain Curiosity test scenes; demonstrates controlled CE+Dice optimization gains.",
          rows: [
            {
              label: "DeepLabV3+ (Baseline)",
              splitNote: "MSL Test 322",
              checkpointType: "Baseline",
              paramsM: dlParams,
              metrics: data.splits.test_msl.deeplabv3plus,
            },
            {
              label: "U-Net (Baseline)",
              splitNote: "MSL Test 322",
              checkpointType: "Baseline",
              paramsM: unetParams,
              metrics: data.splits.test_msl.unet,
            },
            ...(dlOpt
              ? [
                  {
                    label: "DeepLabV3+ (Optimized)",
                    splitNote: "MSL Test 322",
                    checkpointType: "Optimized" as const,
                    paramsM: dlParams,
                    metrics: dlOpt.optimized_test_msl,
                  },
                ]
              : []),
            ...(unetOpt
              ? [
                  {
                    label: "U-Net (Optimized)",
                    splitNote: "MSL Test 322",
                    checkpointType: "Optimized" as const,
                    paramsM: unetParams,
                    metrics: unetOpt.optimized_test_msl,
                  },
                ]
              : []),
          ],
        };
      case "test_mer":
        return {
          title: "Expert Test — MER (Opportunity & Spirit) Subset (204 Images)",
          subtitle: "Out-of-distribution zero-shot evaluation (141 MER-1 Opportunity + 63 MER-2 Spirit).",
          rows: [
            {
              label: "DeepLabV3+ (Baseline)",
              splitNote: "MER Test 204",
              checkpointType: "Baseline",
              paramsM: dlParams,
              metrics: data.splits.test_mer.deeplabv3plus,
            },
            {
              label: "U-Net (Baseline)",
              splitNote: "MER Test 204",
              checkpointType: "Baseline",
              paramsM: unetParams,
              metrics: data.splits.test_mer.unet,
            },
            ...(dlOpt
              ? [
                  {
                    label: "DeepLabV3+ (Optimized)",
                    splitNote: "MER Test 204",
                    checkpointType: "Optimized" as const,
                    paramsM: dlParams,
                    metrics: dlOpt.optimized_test_mer,
                  },
                ]
              : []),
            ...(unetOpt
              ? [
                  {
                    label: "U-Net (Optimized)",
                    splitNote: "MER Test 204",
                    checkpointType: "Optimized" as const,
                    paramsM: unetParams,
                    metrics: unetOpt.optimized_test_mer,
                  },
                ]
              : []),
          ],
        };
      default:
        return {
          title: "Combined JPL Expert Test Set (526 Images, 53,104,279 Valid Pixels)",
          subtitle: "Strictly held out during training; ignore_index=255 excluded from all calculations.",
          rows: [
            {
              label: "DeepLabV3+ (Baseline)",
              splitNote: "Combined 526",
              checkpointType: "Baseline",
              paramsM: dlParams,
              metrics: data.splits.test_combined.deeplabv3plus,
            },
            {
              label: "U-Net (Baseline)",
              splitNote: "Combined 526",
              checkpointType: "Baseline",
              paramsM: unetParams,
              metrics: data.splits.test_combined.unet,
            },
            ...(dlOpt
              ? [
                  {
                    label: "DeepLabV3+ (Optimized)",
                    splitNote: "Combined 526",
                    checkpointType: "Optimized" as const,
                    paramsM: dlParams,
                    metrics: dlOpt.optimized_test_combined,
                  },
                ]
              : []),
            ...(unetOpt
              ? [
                  {
                    label: "U-Net (Optimized)",
                    splitNote: "Combined 526",
                    checkpointType: "Optimized" as const,
                    paramsM: unetParams,
                    metrics: unetOpt.optimized_test_combined,
                  },
                ]
              : []),
          ],
        };
    }
  };

  const cmUrl = resolveArtifactUrl(data.artifacts.confusion_matrices_url);
  const splitData = getRowsForSplit(activeTab);

  // Domain shift delta values computed directly from observed JSON
  const dlMslMiou = data.splits.test_msl.deeplabv3plus.mIoU;
  const dlMerMiou = data.splits.test_mer.deeplabv3plus.mIoU;
  const unetMslMiou = data.splits.test_msl.unet.mIoU;
  const unetMerMiou = data.splits.test_mer.unet.mIoU;

  return (
    <div className="space-y-5">
      {/* Split selector tabs — never mixes splits */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#303431] pb-3">
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Evaluation Splits">
          {[
            { id: "test_combined" as SplitTab, label: "Expert Test · Combined (526)" },
            { id: "test_msl" as SplitTab, label: "Expert Test · MSL (322)" },
            { id: "test_mer" as SplitTab, label: "Expert Test · MER (204)" },
            { id: "validation" as SplitTab, label: "Validation (363)" },
            { id: "benchmarks" as SplitTab, label: "Latency & GPU Benchmarks" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 text-[12px] font-medium rounded-[2px] border transition-colors ${
                activeTab === tab.id
                  ? "bg-[#202321] border-[#C87951] text-[#EAE9E4]"
                  : "bg-[#121414] border-[#303431] text-[#9B9E98] hover:text-[#EAE9E4]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="text-[11px] font-mono-tech text-[#9B9E98]">
          Split Integrity Enforced · Zero Cross-Split Averaging
        </div>
      </div>

      {/* MSL vs MER Domain-Shift Notice Banner */}
      <section className="mission-panel p-4 border-l-2 border-l-[#C87951] grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
        <div className="lg:col-span-7">
          <div className="text-[10px] font-mono-tech uppercase tracking-widest text-[#C87951]">
            Verified Domain-Shift Finding (MSL Curiosity → MER Opportunity / Spirit)
          </div>
          <p className="text-[13px] text-[#EAE9E4] mt-1 leading-relaxed">
            Models trained exclusively on MSL imagery drop{" "}
            <span className="font-mono-tech font-semibold text-[#D0A064]">
              {((dlMslMiou - dlMerMiou) * 100).toFixed(2)} mIoU pts
            </span>{" "}
            (DeepLabV3+) and{" "}
            <span className="font-mono-tech font-semibold text-[#D0A064]">
              {((unetMslMiou - unetMerMiou) * 100).toFixed(2)} mIoU pts
            </span>{" "}
            (U-Net) on MER test frames. Baseline DeepLabV3+ remains the default checkpoint to preserve zero-shot MER
            Big Rock recall while offering Optimized checkpoints for in-domain MSL precision.
          </p>
        </div>

        <div className="lg:col-span-5 grid grid-cols-2 gap-3 font-mono-tech text-[11px]">
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[#696E68]">DEEPLABV3+ MSL → MER</div>
            <div className="text-[15px] text-[#EAE9E4] mt-0.5">
              {dlMslMiou.toFixed(4)} → {dlMerMiou.toFixed(4)}
            </div>
          </div>
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[#696E68]">U-NET MSL → MER</div>
            <div className="text-[15px] text-[#EAE9E4] mt-0.5">
              {unetMslMiou.toFixed(4)} → {unetMerMiou.toFixed(4)}
            </div>
          </div>
        </div>
      </section>

      {activeTab === "benchmarks" ? (
        <FadeContent>
          <section className="mission-panel p-5 space-y-5">
            <div>
              <h2 className="text-[15px] font-semibold text-[#EAE9E4]">
                Hardware &amp; Inference Latency Benchmarks (512×512 RGB)
              </h2>
              <p className="text-[12px] text-[#9B9E98] mt-0.5">
                Measured on {data.benchmarks.environment.gpu_name ?? "CPU"} (
                {data.benchmarks.environment.gpu_vram_gb ?? "?"} GB VRAM) · PyTorch{" "}
                {data.benchmarks.environment.pytorch_version}
              </p>
            </div>

            <div className="table-wrapper" tabIndex={0} role="region" aria-label="Hardware Benchmark Table">
              <table className="w-full text-left text-[12px] border-collapse">
                <thead>
                  <tr className="border-b border-[#303431] bg-[#121414] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                    <th className="py-2.5 px-3">Architecture</th>
                    <th className="py-2.5 px-3">Parameters</th>
                    <th className="py-2.5 px-3">Weights / Ckpt</th>
                    <th className="py-2.5 px-3">GPU FP16 bs=1 Avg / Med</th>
                    <th className="py-2.5 px-3">GPU FP16 FPS</th>
                    <th className="py-2.5 px-3">GPU bs=1 / bs=4 Peak VRAM</th>
                    <th className="py-2.5 px-3">CPU FP32 bs=1 Latency / FPS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#303431] font-mono-tech text-[12px]">
                  {(["deeplabv3plus", "unet"] as const).map((k) => {
                    const m = data.benchmarks.models[k];
                    const fp16b1 = m.profiles.gpu_fp16_bs1;
                    const fp16b4 = m.profiles.gpu_fp16_bs4;
                    const cpub1 = m.profiles.cpu_fp32_bs1;
                    return (
                      <tr key={k} className="hover:bg-[#202321]/60">
                        <td className="py-3 px-3 font-sans font-semibold text-[#EAE9E4]">{m.architecture}</td>
                        <td className="py-3 px-3">{m.parameter_count.toLocaleString()}</td>
                        <td className="py-3 px-3">
                          {m.weights_only_size_mb} / {m.checkpoint_size_mb} MB
                        </td>
                        <td className="py-3 px-3">
                          {fp16b1 ? `${fp16b1.avg_latency_ms} / ${fp16b1.median_latency_ms} ms` : "—"}
                        </td>
                        <td className="py-3 px-3 text-[#829B79]">
                          {fp16b1 ? `${fp16b1.throughput_images_per_sec} img/s` : "—"}
                        </td>
                        <td className="py-3 px-3">
                          {fp16b1 && fp16b4
                            ? `${fp16b1.peak_gpu_memory_allocated_mb} / ${fp16b4.peak_gpu_memory_allocated_mb} MB`
                            : "—"}
                        </td>
                        <td className="py-3 px-3">
                          {cpub1 ? `${cpub1.avg_latency_ms} ms (${cpub1.throughput_images_per_sec} img/s)` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </FadeContent>
      ) : (
        <FadeContent key={activeTab}>
          <section className="mission-panel p-5 space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-[16px] font-semibold text-[#EAE9E4]">{splitData.title}</h2>
                <p className="text-[12px] text-[#9B9E98] mt-0.5">{splitData.subtitle}</p>
              </div>
              <div className="text-[11px] font-mono-tech text-[#696E68]">
                Single-split view only · {splitData.rows[0]?.splitNote}
              </div>
            </div>

            {/* Restrained SVG Comparison Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {splitData.rows.map((row) => (
                <div key={row.label} className="p-3.5 bg-[#121414] border border-[#303431] rounded-[2px] space-y-2.5">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="text-[13px] font-semibold text-[#EAE9E4]">{row.label}</div>
                      <div className="text-[10px] font-mono-tech text-[#9B9E98]">
                        {row.checkpointType} Checkpoint · {row.paramsM} params
                      </div>
                    </div>
                    <div className="text-right font-mono-tech">
                      <div className="text-[16px] font-semibold text-[#EAE9E4]">{row.metrics.mIoU.toFixed(4)}</div>
                      <div className="text-[10px] text-[#696E68]">mIoU</div>
                    </div>
                  </div>

                  {/* Per-class IoU horizontal bars */}
                  <div className="space-y-1.5 text-[11px]">
                    {[
                      { label: "Soil", iou: row.metrics.per_class_iou.soil, hex: TERRAIN_CLASSES[0].hex },
                      { label: "Bedrock", iou: row.metrics.per_class_iou.bedrock, hex: TERRAIN_CLASSES[1].hex },
                      { label: "Sand", iou: row.metrics.per_class_iou.sand, hex: TERRAIN_CLASSES[2].hex },
                      { label: "Big Rock", iou: row.metrics.per_class_iou["big rock"], hex: TERRAIN_CLASSES[3].hex },
                    ].map((cls) => (
                      <div key={cls.label} className="flex items-center gap-2">
                        <span className="w-16 text-[#9B9E98]">{cls.label}</span>
                        <div className="flex-1 h-1.5 bg-[#181A19]">
                          <div className="h-full" style={{ width: `${cls.iou * 100}%`, backgroundColor: cls.hex }} />
                        </div>
                        <span className="w-14 text-right font-mono-tech text-[#EAE9E4]">
                          {(cls.iou * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Detailed Precision / Recall / IoU Metrics Table */}
            <div className="table-wrapper" tabIndex={0} role="region" aria-label="Per-Model Metrics Comparison Table">
              <table className="w-full text-left text-[12px] border-collapse">
                <thead>
                  <tr className="border-b border-[#303431] bg-[#121414] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-2.5">Type</th>
                    <th className="py-2.5 px-2.5">mIoU</th>
                    <th className="py-2.5 px-2.5">Mean Dice</th>
                    <th className="py-2.5 px-2.5">Pixel Acc</th>
                    <th className="py-2.5 px-2.5">Soil IoU</th>
                    <th className="py-2.5 px-2.5">Bedrock IoU</th>
                    <th className="py-2.5 px-2.5">Sand IoU</th>
                    <th className="py-2.5 px-2.5">Big Rock IoU</th>
                    <th className="py-2.5 px-2.5">Big Rock Prec / Rec</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#303431] font-mono-tech text-[11px]">
                  {splitData.rows.map((row) => (
                    <tr key={row.label} className="hover:bg-[#202321]/60">
                      <td className="py-2.5 px-3 font-sans font-medium text-[#EAE9E4]">{row.label}</td>
                      <td className="py-2.5 px-2.5 text-[#9B9E98]">{row.checkpointType}</td>
                      <td className="py-2.5 px-2.5 font-semibold text-[#EAE9E4]">{row.metrics.mIoU.toFixed(4)}</td>
                      <td className="py-2.5 px-2.5">{row.metrics.mean_dice.toFixed(4)}</td>
                      <td className="py-2.5 px-2.5">{row.metrics.pixel_accuracy.toFixed(4)}</td>
                      <td className="py-2.5 px-2.5">{row.metrics.per_class_iou.soil.toFixed(4)}</td>
                      <td className="py-2.5 px-2.5">{row.metrics.per_class_iou.bedrock.toFixed(4)}</td>
                      <td className="py-2.5 px-2.5">{row.metrics.per_class_iou.sand.toFixed(4)}</td>
                      <td className="py-2.5 px-2.5 font-semibold text-[#EAE9E4]">
                        {row.metrics.per_class_iou["big rock"].toFixed(4)}
                      </td>
                      <td className="py-2.5 px-2.5">
                        {row.metrics.per_class_precision["big rock"].toFixed(3)} /{" "}
                        {row.metrics.per_class_recall["big rock"].toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* MER Sub-mission breakdown if on MER tab */}
            {activeTab === "test_mer" && data.splits.test_mer.mer_submissions && (
              <div className="pt-2 border-t border-[#303431]">
                <h3 className="text-[12px] font-mono-tech uppercase text-[#9B9E98] mb-3">
                  MER Sub-Mission Breakdown (Baseline Models)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 font-mono-tech text-[11px]">
                  {Object.entries(data.splits.test_mer.mer_submissions).map(([subName, subData]) => (
                    <div key={subName} className="p-3 bg-[#121414] border border-[#303431] space-y-1.5">
                      <div className="font-sans font-semibold text-[13px] text-[#EAE9E4]">
                        {subName} ({subData.num_images} images)
                      </div>
                      <div className="flex justify-between text-[#9B9E98]">
                        <span>DeepLabV3+ mIoU / Bedrock IoU</span>
                        <span className="text-[#EAE9E4]">
                          {subData.deeplabv3plus.mIoU.toFixed(4)} / {subData.deeplabv3plus.per_class_iou.bedrock.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between text-[#9B9E98]">
                        <span>U-Net mIoU / Bedrock IoU</span>
                        <span className="text-[#EAE9E4]">
                          {subData.unet.mIoU.toFixed(4)} / {subData.unet.per_class_iou.bedrock.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </FadeContent>
      )}

      {/* Confusion Matrices Reference Artifact */}
      {cmUrl && (
        <section className="mission-panel p-4">
          <div className="flex items-center justify-between gap-2 pb-3 mb-3 border-b border-[#303431]">
            <h2 className="text-[13px] font-semibold uppercase text-[#EAE9E4] flex items-center gap-2">
              <Icon name="table-cells-large" className="text-[#C87951] text-[12px]" />
              <span>Expert Test Row-Normalized Confusion Matrices (Baseline Models)</span>
            </h2>
            <span className="text-[11px] font-mono-tech text-[#696E68]">Combined / MSL / MER</span>
          </div>
          <img
            src={cmUrl}
            alt="Expert Test Row-Normalized Confusion Matrices"
            className="w-full max-w-[1150px] mx-auto block border border-[#303431]"
          />
        </section>
      )}
    </div>
  );
};
