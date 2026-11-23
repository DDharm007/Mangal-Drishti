import type { TerrainKey, TraversabilityAssessment } from "../types/api";

export interface TerrainVisualMeta {
  key: TerrainKey;
  id: number;
  label: string;
  hex: string;
  rgb: [number, number, number];
  faIcon: string;
}

// Exact RGB & hex colors derived strictly from datasets/metadata/class_mapping.json
export const TERRAIN_CLASSES: TerrainVisualMeta[] = [
  {
    key: "soil",
    id: 0,
    label: "Soil",
    hex: "#A6611A",
    rgb: [166, 97, 26],
    faIcon: "layer-group",
  },
  {
    key: "bedrock",
    id: 1,
    label: "Bedrock",
    hex: "#377EB8",
    rgb: [55, 126, 184],
    faIcon: "mountain",
  },
  {
    key: "sand",
    id: 2,
    label: "Sand",
    hex: "#F5C242",
    rgb: [245, 194, 66],
    faIcon: "wind",
  },
  {
    key: "big_rock",
    id: 3,
    label: "Big Rock",
    hex: "#E41A1C",
    rgb: [228, 26, 28],
    faIcon: "cubes",
  },
];

export const IGNORE_CLASS_META = {
  key: "ignore",
  id: 255,
  label: "Ignore / Unlabeled (255)",
  hex: "#282C34",
  rgb: [40, 44, 52] as [number, number, number],
};

export function getTerrainMeta(key: string | null | undefined): TerrainVisualMeta | null {
  if (!key) return null;
  const norm = key.toLowerCase().replace(" ", "_");
  return TERRAIN_CLASSES.find((c) => c.key === norm) ?? null;
}

export function formatConcernLabel(assessment: TraversabilityAssessment | string | null | undefined): {
  label: string;
  toneClass: string;
  borderClass: string;
} {
  switch (assessment) {
    case "lower_visual_concern":
      return {
        label: "Lower Visual Concern",
        toneClass: "text-[#829B79]",
        borderClass: "border-[#829B79]/40 bg-[#829B79]/10",
      };
    case "moderate_visual_concern":
      return {
        label: "Moderate Visual Concern",
        toneClass: "text-[#D0A064]",
        borderClass: "border-[#D0A064]/40 bg-[#D0A064]/10",
      };
    case "higher_visual_concern":
      return {
        label: "Higher Visual Concern",
        toneClass: "text-[#C76E64]",
        borderClass: "border-[#C76E64]/45 bg-[#C76E64]/10",
      };
    default:
      return {
        label: "Insufficient Visual Evidence",
        toneClass: "text-[#9B9E98]",
        borderClass: "border-[#303431] bg-[#202321]",
      };
  }
}

export function formatUtcTimestamp(ts: string | null | undefined): string {
  if (!ts) return "Unavailable";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

export function formatModelName(model: string): string {
  switch (model) {
    case "deeplabv3plus":
      return "DeepLabV3+ (Baseline)";
    case "unet":
      return "U-Net (Baseline)";
    case "deeplabv3plus_optimized":
      return "DeepLabV3+ (Optimized)";
    case "unet_optimized":
      return "U-Net (Optimized)";
    default:
      return model;
  }
}
