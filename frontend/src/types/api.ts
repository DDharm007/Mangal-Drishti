export type TerrainKey = "soil" | "bedrock" | "sand" | "big_rock";

export type TraversabilityAssessment =
  | "lower_visual_concern"
  | "moderate_visual_concern"
  | "higher_visual_concern"
  | "insufficient_visual_evidence";

export interface ArtifactUrls {
  original_image: string | null;
  segmentation_mask: string | null;
  terrain_overlay: string | null;
  region_boundaries: string | null;
  big_rock_candidates: string | null;
  confidence_heatmap: string | null;
  uncertainty_heatmap: string | null;
  terrain_distribution_chart: string | null;
  intelligence_summary_panel: string | null;
  intelligence_report_json: string | null;
}

export interface RegionItem {
  region_id: string;
  class_id: number;
  terrain_class: TerrainKey;
  pixel_area: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  contour: {
    num_vertices: number;
    simplified_vertices: [number, number][];
    perimeter_px: number;
  };
  relative_position: string;
  mean_confidence: number;
  image_coverage_pct: number;
}

export interface ObstacleCandidate {
  candidate_id: string;
  source_class: string;
  bbox: [number, number, number, number];
  pixel_area: number;
  image_coverage_pct: number;
  confidence: number;
  confidence_details: {
    mean_big_rock_probability: number;
    max_big_rock_probability: number;
    mean_softmax_confidence: number;
  };
  centroid: [number, number];
  shape: {
    perimeter_px: number;
    aspect_ratio_w_over_h: number;
    circularity: number;
    solidity: number;
    bbox_extent: number;
  };
  relative_position: string;
  is_confirmed_physical_obstacle: false;
  scientific_note: string;
}

export interface TerrainIntelligenceReport {
  project: string;
  schema_version: string;
  generated_at_utc?: string;
  model: string;
  checkpoint: string;
  device: string;
  image_metadata: {
    image_id: string;
    source_path: string;
    width: number;
    height: number;
    inference_latency_ms: number | null;
  };
  analysis: {
    terrain_summary: Record<TerrainKey, number>;
    terrain_details: {
      total_pixels: number;
      total_valid_pixels: number;
      ignored_pixels: number;
      valid_pixel_percentage: number;
      class_pixel_counts: Record<TerrainKey, number>;
      dominant_terrain: TerrainKey | null;
      secondary_terrain: TerrainKey | null;
      terrain_diversity_shannon_nats: number;
      terrain_diversity_normalized: number;
      boundary_pixel_count: number;
      boundary_density_pct: number;
    };
    regions: RegionItem[];
    region_filtering?: {
      min_region_area_px: number;
      retained_region_count: number;
      filtered_small_component_count: number;
    };
    obstacle_candidates: ObstacleCandidate[];
    confidence: {
      mean_confidence: number;
      median_confidence: number;
      min_confidence: number;
      max_confidence: number;
      std_confidence: number;
      low_confidence_pixel_percentage: number;
      per_class_mean_confidence: Record<TerrainKey, number | null>;
      confidence_categories_pct: {
        high: number;
        medium: number;
        low: number;
      };
      thresholds: {
        high_min: number;
        medium_min: number;
      };
      confidence_histogram: {
        bin_edges: number[];
        pixel_counts: number[];
      };
      is_calibrated_probability: false;
      calibration_note: string;
    };
    uncertainty: {
      entropy_formula: string;
      max_theoretical_entropy_nats: number;
      mean_entropy: number;
      median_entropy: number;
      std_entropy: number;
      max_entropy: number;
      mean_normalized_entropy: number;
      high_uncertainty_pixel_percentage: number;
      high_uncertainty_normalized_threshold: number;
      per_class_mean_entropy: Record<TerrainKey, number | null>;
    };
    traversability: {
      assessment: TraversabilityAssessment;
      experimental_visual_concern_index: number;
      is_experimental: true;
      physical_safety_verified: false;
      insufficient_evidence_triggers: string[];
      component_scores: {
        terrain_composition_score: number;
        big_rock_candidate_density_score: number;
        sand_concentration_score: number;
        segmentation_uncertainty_score: number;
        region_fragmentation_score: number;
      };
      terrain_class_indicators: {
        soil_pct: number;
        bedrock_pct: number;
        sand_pct: number;
        big_rock_pct: number;
        big_rock_candidate_count: number;
        retained_region_count: number;
      };
      missing_measurements: string[];
      formula_summary: string;
      scientific_disclaimer: string;
    };
  };
  artifacts: Record<string, string>;
}

export interface ReportSummary {
  report_id: string;
  image_id: string;
  mission: string | null;
  model: string;
  device: string;
  generated_at_utc: string;
  dominant_terrain: TerrainKey | null;
  secondary_terrain: TerrainKey | null;
  valid_pixel_percentage: number;
  terrain_summary: Record<TerrainKey, number>;
  retained_regions: number;
  obstacle_candidates_count: number;
  mean_confidence: number;
  mean_entropy: number;
  traversability_assessment: TraversabilityAssessment;
  experimental_visual_concern_index: number;
  physical_safety_verified: boolean;
  artifact_urls: ArtifactUrls;
}

export interface ReportDetailResponse {
  report_id: string;
  mission: string | null;
  artifact_urls: ArtifactUrls;
  report: TerrainIntelligenceReport;
}

export interface CheckpointStatus {
  model_name: string;
  label: string;
  checkpoint_path: string;
  available: boolean;
  size_mb: number | null;
  loaded_in_memory: boolean;
}

export interface SystemStatusResponse {
  status: "online" | "offline" | "degraded";
  version: string;
  pytorch_version: string;
  cuda_available: boolean;
  active_device: string;
  gpu_name: string | null;
  gpu_vram_gb: number | null;
  host_ram_gb: number;
  checkpoints: CheckpointStatus[];
  loaded_models: string[];
  stored_reports_count: number;
  latest_analysis_id: string | null;
  latest_analysis_timestamp_utc: string | null;
  report_storage_dir: string;
}

export interface ModelMetricsBlock {
  mIoU: number;
  mean_dice: number;
  pixel_accuracy: number;
  per_class_iou: { soil: number; bedrock: number; sand: number; "big rock": number };
  per_class_dice: { soil: number; bedrock: number; sand: number; "big rock": number };
  per_class_precision: { soil: number; bedrock: number; sand: number; "big rock": number };
  per_class_recall: { soil: number; bedrock: number; sand: number; "big rock": number };
  valid_pixels_evaluated: number;
  num_images?: number;
  confusion_matrix_labels?: string[];
  confusion_matrix?: number[][];
  confusion_matrix_row_normalized?: number[][];
}

export interface ModelsPerformanceResponse {
  catalog: Array<{
    id: string;
    label: string;
    backbone: string;
    status_note: string;
    is_default: boolean;
    checkpoint_available: boolean;
    loaded_in_memory: boolean;
  }>;
  splits: {
    validation: any;
    test_combined: {
      num_images: number;
      valid_pixels_evaluated: number;
      unet: ModelMetricsBlock;
      deeplabv3plus: ModelMetricsBlock;
      comparison_deeplabv3plus_minus_unet: Record<string, number>;
    };
    test_msl: {
      num_images: number;
      valid_pixels_evaluated: number;
      unet: ModelMetricsBlock;
      deeplabv3plus: ModelMetricsBlock;
      comparison_deeplabv3plus_minus_unet: Record<string, number>;
    };
    test_mer: {
      num_images: number;
      valid_pixels_evaluated: number;
      unet: ModelMetricsBlock;
      deeplabv3plus: ModelMetricsBlock;
      comparison_deeplabv3plus_minus_unet: Record<string, number>;
      mer_submissions: Record<string, { num_images: number; unet: ModelMetricsBlock; deeplabv3plus: ModelMetricsBlock }>;
    };
  };
  benchmarks: {
    environment: {
      gpu_name: string | null;
      gpu_vram_gb: number | null;
      ram_gb: number;
      cpu_threads_used: number;
      pytorch_version: string;
      input_resolution: string;
    };
    models: Record<
      "unet" | "deeplabv3plus",
      {
        architecture: string;
        parameter_count: number;
        weights_only_size_mb: number;
        checkpoint_size_mb: number;
        profiles: Record<
          string,
          {
            device: string;
            precision: string;
            batch_size: number;
            avg_latency_ms: number;
            median_latency_ms: number;
            p95_latency_ms: number;
            throughput_images_per_sec: number;
            peak_gpu_memory_allocated_mb: number | null;
          }
        >;
      }
    >;
  };
  optimization: {
    protocol: string;
    experiments: Record<
      string,
      {
        baseline_val: ModelMetricsBlock;
        optimized_val: ModelMetricsBlock & { best_epoch: number };
        baseline_test_combined: ModelMetricsBlock;
        optimized_test_combined: ModelMetricsBlock;
        optimized_test_msl: ModelMetricsBlock;
        optimized_test_mer: ModelMetricsBlock;
      }
    >;
  };
  artifacts: {
    confusion_matrices_url: string | null;
  };
}

export interface ResearchResponse {
  class_mapping: {
    dataset: string;
    active_taxonomy: string;
    num_trainable_classes: number;
    ignore_index: number;
    mask_interpolation_rule: string;
    classes: Array<{
      original_id: number;
      class_name: string;
      trainable: boolean;
      ignored: boolean;
      palette_rgb: [number, number, number];
    }>;
  };
  class_weights: {
    train_imbalance_ratio_max_to_min: number;
    recommended_default_method: string;
    weighting_methods: Record<string, Record<string, number>>;
    split_pixel_distributions: Record<string, any>;
  };
  split_summary: {
    dataset: string;
    random_seed: number;
    split_config: Record<string, any>;
    counts: Record<string, any>;
    leakage_checks: Record<string, number>;
  };
  documentation: Array<{
    filename: string;
    title: string;
    path: string;
    content: string;
  }>;
}
