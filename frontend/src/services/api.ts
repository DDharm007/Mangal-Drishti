import type {
  ModelsPerformanceResponse,
  ReportDetailResponse,
  ReportSummary,
  ResearchResponse,
  SystemStatusResponse,
} from "../types/api";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) || "";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody?.detail) {
        detail = errBody.detail;
      }
    } catch {
      // Ignore non-JSON error responses
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function fetchHealth(): Promise<{
  status: string;
  service: string;
  version: string;
  timestamp_utc: string;
}> {
  return requestJson("/api/health");
}

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  return requestJson("/api/system/status");
}

export async function fetchReports(): Promise<{ count: number; reports: ReportSummary[] }> {
  return requestJson("/api/reports");
}

export async function fetchReportDetail(reportId: string): Promise<ReportDetailResponse> {
  return requestJson(`/api/reports/${encodeURIComponent(reportId)}`);
}

export async function analyzeRoverImage(
  file: File,
  modelName: string
): Promise<ReportDetailResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", modelName);
  return requestJson("/api/analyze", {
    method: "POST",
    body: form,
  });
}

export async function fetchModelsPerformance(): Promise<ModelsPerformanceResponse> {
  return requestJson("/api/models");
}

export async function fetchResearch(): Promise<ResearchResponse> {
  return requestJson("/api/research");
}

export function getDownloadReportUrl(reportId: string): string {
  return `${API_BASE}/api/reports/${encodeURIComponent(reportId)}/download`;
}

export function resolveArtifactUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path}`;
}
