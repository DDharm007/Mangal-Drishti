import React, { useEffect, useMemo, useState } from "react";
import { MarkdownRenderer, extractHeadings } from "../components/research/MarkdownRenderer";
import { Icon } from "../components/ui/Icon";
import { fetchResearch } from "../services/api";
import type { ResearchResponse } from "../types/api";

export const DatasetResearchPage: React.FC = () => {
  const [data, setData] = useState<ResearchResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocIdx, setSelectedDocIdx] = useState(0);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        setLoading(true);
        const res = await fetchResearch();
        if (active) setData(res);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : "Failed to load research metadata.");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const activeDoc = data?.documentation?.[selectedDocIdx] ?? null;
  const headings = useMemo(
    () => (activeDoc ? extractHeadings(activeDoc.content).filter((h) => h.level <= 2) : []),
    [activeDoc]
  );

  if (loading) {
    return (
      <div className="mission-panel p-10 text-center font-mono-tech text-[13px] text-[#9B9E98]">
        Loading NASA AI4MARS dataset metadata and research documentation...
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="p-4 border border-[#C76E64]/45 bg-[#C76E64]/10 text-[13px] text-[#EAE9E4]">
        {error ?? "Dataset metadata unavailable."}
      </div>
    );
  }

  const counts = data.split_summary?.counts ?? {};
  const leakage = data.split_summary?.leakage_checks ?? {};

  return (
    <div className="space-y-4">
      {/* Dataset Source & Split Integrity Summary */}
      <section className="mission-panel p-4 space-y-3.5">
        <div className="flex flex-wrap items-start justify-between gap-3 pb-3 border-b border-[#303431]">
          <div>
            <div className="text-[10px] font-mono-tech uppercase tracking-widest text-[#C87951]">
              Primary Corpus · Official NASA / JPL Annotation Taxonomy
            </div>
            <h2 className="text-[17px] font-semibold text-[#EAE9E4] mt-0.5">
              {data.class_mapping.dataset} — NAV 4-Class Split &amp; Engineering Provenance
            </h2>
          </div>
          <a
            href="https://zenodo.org/records/15995036"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#121414] border border-[#303431] hover:border-[#C87951] text-[11px] font-mono-tech text-[#EAE9E4] rounded-[2px]"
          >
            <span>Zenodo AI4MARS Record 15995036</span>
            <Icon name="arrow-up-right-from-square" className="text-[10px]" />
          </a>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 font-mono-tech text-[12px]">
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">RAW / VALID PAIRS</div>
            <div className="text-[#EAE9E4] font-semibold mt-0.5">
              {counts.raw_total_pairs ?? "—"} / {counts.valid_total_pairs ?? "—"}
            </div>
          </div>
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">TRAIN (MSL SCLK)</div>
            <div className="text-[#EAE9E4] font-semibold mt-0.5">{counts.train ?? "—"}</div>
          </div>
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">VALIDATION (MSL SCLK)</div>
            <div className="text-[#EAE9E4] font-semibold mt-0.5">{counts.val ?? "—"}</div>
          </div>
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">EXPERT TEST (MSL+MER)</div>
            <div className="text-[#EAE9E4] font-semibold mt-0.5">{counts.test ?? "—"}</div>
          </div>
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">REJECTED ALL-255</div>
            <div className="text-[#EAE9E4] font-semibold mt-0.5">{counts.rejected_pairs ?? "—"}</div>
          </div>
          <div className="p-2.5 bg-[#121414] border border-[#303431]">
            <div className="text-[10px] text-[#696E68]">SPLIT LEAKAGE</div>
            <div className="text-[#829B79] font-semibold mt-0.5">
              {leakage.train_val_sclk_cluster_overlap ?? 0} overlaps
            </div>
          </div>
        </div>
      </section>

      {/* Class Mapping, Colors & Architecture Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <section className="lg:col-span-6 mission-panel p-4 space-y-3">
          <h3 className="text-[12.5px] font-mono-tech font-semibold uppercase text-[#EAE9E4] pb-2 border-b border-[#303431]">
            Verified Class Encoding &amp; Ignore Index {data.class_mapping.ignore_index}
          </h3>
          <div className="table-wrapper" tabIndex={0} role="region" aria-label="Terrain Class Mapping">
            <table className="w-full text-left text-[12px] border-collapse">
              <thead>
                <tr className="border-b border-[#303431] text-[10px] font-mono-tech uppercase text-[#9B9E98]">
                  <th className="py-2 pr-3">ID</th>
                  <th className="py-2 pr-3">Class</th>
                  <th className="py-2 pr-3">Step 2 RGB</th>
                  <th className="py-2 pr-3">Train Pixel %</th>
                  <th className="py-2">Weight (Med. Freq)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#303431]/70 font-mono-tech text-[11px]">
                {data.class_mapping.classes.map((cls) => {
                  const rgbStr = `rgb(${cls.palette_rgb.join(", ")})`;
                  const trainCls =
                    data.class_weights?.split_pixel_distributions?.train?.classes?.[String(cls.original_id)];
                  const wt = data.class_weights?.weighting_methods?.median_frequency_capped?.[String(cls.original_id)];
                  return (
                    <tr key={cls.original_id}>
                      <td className="py-2 pr-3 text-[#EAE9E4]">{cls.original_id}</td>
                      <td className="py-2 pr-3 font-sans font-medium capitalize text-[#EAE9E4]">
                        {cls.class_name.replace("_", " ")}
                      </td>
                      <td className="py-2 pr-3">
                        <span className="inline-flex items-center gap-2">
                          <span className="w-3 h-3 rounded-[1px]" style={{ backgroundColor: rgbStr }} />
                          <span>[{cls.palette_rgb.join(",")}]</span>
                        </span>
                      </td>
                      <td className="py-2 pr-3">
                        {trainCls ? `${trainCls.pct_of_valid_pixels}%` : "Excluded"}
                      </td>
                      <td className="py-2">{wt !== undefined ? wt : "Ignored"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[12px] text-[#9B9E98] leading-relaxed">
            <strong className="text-[#EAE9E4]">Mask Integrity Rule:</strong> {data.class_mapping.mask_interpolation_rule}.
            Pixels matching <code className="font-mono-tech text-[#EAE9E4]">255</code> encode rover hardware masks (MXY),
            &gt;30 m stereo range masks (RNG-30M), and unlabeled regions, excluded from loss and metrics.
          </p>
        </section>

        <section className="lg:col-span-6 mission-panel p-4 space-y-3 text-[12.5px] text-[#9B9E98] leading-relaxed">
          <h3 className="text-[12.5px] font-mono-tech font-semibold uppercase text-[#EAE9E4] pb-2 border-b border-[#303431]">
            Architecture, MSL vs. MER Distinction &amp; Scientific Boundaries
          </h3>
          <ul className="space-y-2 list-disc pl-5">
            <li>
              <strong className="text-[#EAE9E4]">Temporal Drive Split (Zero Leakage):</strong> 1,770 non-empty MSL frames
              are grouped into 189 SCLK temporal drive clusters (43,200 s gap) and split 153 train / 36 validation.
            </li>
            <li>
              <strong className="text-[#EAE9E4]">Held-Out Expert Gold Standard:</strong> 526 expert consensus frames (322
              MSL Curiosity + 141 MER-1 Opportunity + 63 MER-2 Spirit) remain strictly isolated from training.
            </li>
            <li>
              <strong className="text-[#EAE9E4]">Segmentation Backbones:</strong> U-Net + ResNet34 (
              <code className="font-mono-tech text-[#EAE9E4]">24.52M</code> params, skip-connection decoder) and
              DeepLabV3+ + ResNet50 (<code className="font-mono-tech text-[#EAE9E4]">40.35M</code> params, ASPP +
              low-level fusion).
            </li>
            <li>
              <strong className="text-[#EAE9E4]">Non-Traversability Limitation:</strong> 2D pixel terrain classes do not
              measure stereo elevation, terrain slope, obstacle height, or wheel slip mechanics.
            </li>
          </ul>
        </section>
      </div>

      {/* Rendered Research Documentation Workspace (TOC Nav + Formatted Markdown Reader) */}
      {data.documentation.length > 0 && (
        <section className="mission-panel grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
          <aside className="lg:col-span-3 border-b lg:border-b-0 lg:border-r border-[#303431] bg-[#121414] p-3.5 space-y-4">
            <div>
              <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68] mb-2">
                Repository Documents
              </div>
              <div className="space-y-1">
                {data.documentation.map((doc, i) => (
                  <button
                    key={doc.filename}
                    type="button"
                    onClick={() => setSelectedDocIdx(i)}
                    className={`w-full text-left px-2.5 py-2 rounded-[2px] border text-[11.5px] transition-colors ${
                      selectedDocIdx === i
                        ? "bg-[#202321] border-[#C87951] text-[#EAE9E4]"
                        : "border-transparent text-[#9B9E98] hover:text-[#EAE9E4] hover:bg-[#181A19]"
                    }`}
                  >
                    <div className="font-medium truncate">{doc.title}</div>
                    <div className="text-[10px] font-mono-tech text-[#696E68] truncate">{doc.path}</div>
                  </button>
                ))}
              </div>
            </div>

            {headings.length > 0 && (
              <div className="pt-3 border-t border-[#303431]">
                <div className="text-[10px] font-mono-tech uppercase tracking-wider text-[#696E68] mb-2">
                  Section Outline
                </div>
                <ul className="space-y-1 text-[11px] max-h-[300px] overflow-y-auto pr-1">
                  {headings.map((h) => (
                    <li key={h.id}>
                      <a
                        href={`#${h.id}`}
                        className="block py-1 text-[#9B9E98] hover:text-[#C87951] truncate transition-colors"
                      >
                        {h.text}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>

          <div className="lg:col-span-9 p-5 sm:p-6 bg-[#0D0E0E] max-h-[700px] overflow-y-auto">
            {activeDoc && <MarkdownRenderer content={activeDoc.content} />}
          </div>
        </section>
      )}
    </div>
  );
};
