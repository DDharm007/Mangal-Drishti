import React, { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import type { SystemStatusResponse } from "../../types/api";
import { Icon } from "../ui/Icon";

interface NavItem {
  to: string;
  label: string;
  icon: string;
  title: string;
  breadcrumb: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: "satellite-dish", title: "Mission Overview", breadcrumb: "Control / Overview" },
  {
    to: "/analyze",
    label: "Terrain Analysis",
    icon: "crosshairs",
    title: "Terrain Analysis Workspace",
    breadcrumb: "Control / Analysis",
  },
  {
    to: "/reports",
    label: "Intelligence Reports",
    icon: "folder-open",
    title: "Intelligence Reports",
    breadcrumb: "Archive / Reports",
  },
  {
    to: "/models",
    label: "Model Performance",
    icon: "chart-column",
    title: "Model Benchmarks & Performance",
    breadcrumb: "Evaluation / Models",
  },
  {
    to: "/research",
    label: "Dataset & Research",
    icon: "book-open",
    title: "Dataset & Scientific Research",
    breadcrumb: "Documentation / AI4MARS",
  },
  {
    to: "/status",
    label: "System Status",
    icon: "microchip",
    title: "System Diagnostics",
    breadcrumb: "Operations / Status",
  },
];

export interface AppShellProps {
  children: React.ReactNode;
  systemStatus: SystemStatusResponse | null;
  statusError: string | null;
  activeModel: string;
}

const BrandLogo: React.FC = () => (
  <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true" className="shrink-0">
    <circle cx="14" cy="14" r="12" stroke="#C87951" strokeWidth="1.25" />
    <circle cx="14" cy="14" r="5" stroke="#EAE9E4" strokeWidth="1.15" />
    <line x1="14" y1="0" x2="14" y2="6" stroke="#9B9E98" strokeWidth="1.1" />
    <line x1="14" y1="22" x2="14" y2="28" stroke="#9B9E98" strokeWidth="1.1" />
    <line x1="0" y1="14" x2="6" y2="14" stroke="#9B9E98" strokeWidth="1.1" />
    <line x1="22" y1="14" x2="28" y2="14" stroke="#9B9E98" strokeWidth="1.1" />
  </svg>
);

export const AppShell: React.FC<AppShellProps> = ({
  children,
  systemStatus,
  statusError,
  activeModel,
}) => {
  const location = useLocation();
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    setDrawerOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const currentNav =
    NAV_ITEMS.find((n) => (n.to === "/" ? location.pathname === "/" : location.pathname.startsWith(n.to))) ||
    NAV_ITEMS[0];

  const isOnline = systemStatus?.status === "online" && !statusError;
  const deviceLabel = systemStatus
    ? systemStatus.cuda_available && systemStatus.gpu_name
      ? `CUDA · ${systemStatus.gpu_name.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")}`
      : `CPU · ${systemStatus.active_device.toUpperCase()}`
    : "Connecting...";

  const sidebarContent = (
    <div className="flex h-full flex-col justify-between bg-[#121414] border-r border-[#303431]">
      <div>
        <div className="flex items-center gap-3 px-4 py-4 border-b border-[#303431]">
          <BrandLogo />
          <div className="min-w-0">
            <div className="text-[13px] font-semibold tracking-[0.12em] text-[#EAE9E4] uppercase truncate">
              MANGAL DRISHTI
            </div>
            <div className="text-[11px] font-mono-tech text-[#9B9E98] truncate">
              Mars Terrain Intelligence
            </div>
          </div>
        </div>

        <nav className="p-2 space-y-0.5" aria-label="Primary Mission Navigation">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 text-[13px] font-medium rounded-[2px] border-l-2 transition-colors ${
                  isActive
                    ? "bg-[#181A19] text-[#EAE9E4] border-[#C87951]"
                    : "border-transparent text-[#9B9E98] hover:text-[#EAE9E4] hover:bg-[#181A19]/60"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    name={item.icon}
                    className={`w-4 text-center text-[13px] ${isActive ? "text-[#C87951]" : "text-[#696E68]"}`}
                  />
                  <span>{item.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="p-3.5 border-t border-[#303431] space-y-2.5 text-[11px]">
        <div className="space-y-1.5 font-mono-tech">
          <div className="flex items-center justify-between gap-2 text-[#9B9E98]">
            <span>DEVICE</span>
            <span className="text-[#EAE9E4] truncate max-w-[140px]" title={deviceLabel}>
              {deviceLabel}
            </span>
          </div>
          <div className="flex items-center justify-between gap-2 text-[#9B9E98]">
            <span>API LINK</span>
            <span className={`inline-flex items-center gap-1.5 ${isOnline ? "text-[#829B79]" : "text-[#C76E64]"}`}>
              <span
                className={`inline-block w-1.5 h-1.5 rounded-full ${isOnline ? "bg-[#829B79]" : "bg-[#C76E64]"}`}
              />
              {isOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </div>

        <div className="pt-2 border-t border-[#303431] flex items-center justify-between text-[#696E68] font-mono-tech">
          <Link to="/status" className="inline-flex items-center gap-1.5 hover:text-[#EAE9E4] transition-colors">
            <Icon name="sliders" className="text-[11px]" />
            <span>Diagnostics</span>
          </Link>
          <span>v{systemStatus?.version ?? "1.0.0"}</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex bg-[#0D0E0E] text-[#EAE9E4]">
      {/* Fixed desktop sidebar */}
      <aside className="hidden lg:block w-[248px] shrink-0 sticky top-0 h-screen">
        {sidebarContent}
      </aside>

      {/* Mobile / tablet slide-over drawer */}
      {drawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex" role="dialog" aria-modal="true" aria-label="Navigation Menu">
          <div
            className="fixed inset-0 bg-black/70"
            onClick={() => setDrawerOpen(false)}
            aria-hidden="true"
          />
          <aside className="relative z-10 w-[264px] max-w-[82vw] h-full">
            <div className="absolute right-2 top-3 z-20">
              <button
                type="button"
                onClick={() => setDrawerOpen(false)}
                className="p-2 text-[#9B9E98] hover:text-[#EAE9E4]"
                aria-label="Close navigation menu"
              >
                <Icon name="xmark" className="text-base" />
              </button>
            </div>
            {sidebarContent}
          </aside>
        </div>
      )}

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="sticky top-0 z-30 h-14 bg-[#0D0E0E]/95 border-b border-[#303431] px-4 sm:px-6 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="lg:hidden p-2 -ml-2 text-[#9B9E98] hover:text-[#EAE9E4] border border-transparent hover:border-[#303431] rounded-[2px]"
              aria-label="Open navigation menu"
            >
              <Icon name="bars" />
            </button>
            <div className="min-w-0">
              <div className="text-[10px] font-mono-tech uppercase tracking-[0.14em] text-[#696E68] truncate">
                {currentNav.breadcrumb}
              </div>
              <div className="text-[14px] font-semibold text-[#EAE9E4] truncate">
                {currentNav.title}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <div
              className="hidden sm:flex items-center gap-2 px-2.5 py-1 bg-[#181A19] border border-[#303431] rounded-[2px] text-[11px] font-mono-tech text-[#9B9E98]"
              title="Default or selected inference architecture"
            >
              <Icon name="diagram-project" className="text-[#C87951] text-[11px]" />
              <span className="uppercase">{activeModel}</span>
            </div>

            <Link
              to="/status"
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 border rounded-[2px] text-[11px] font-mono-tech ${
                isOnline
                  ? "border-[#829B79]/35 bg-[#829B79]/10 text-[#829B79]"
                  : "border-[#C76E64]/40 bg-[#C76E64]/10 text-[#C76E64]"
              }`}
              title={statusError ?? "Backend telemetry active"}
            >
              <Icon name={isOnline ? "circle-check" : "triangle-exclamation"} className="text-[11px]" />
              <span className="hidden xs:inline">{isOnline ? "API Online" : "API Offline"}</span>
            </Link>

            <Link
              to="/analyze"
              className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#C87951] hover:bg-[#A65E40] text-[#0D0E0E] font-medium text-[12px] rounded-[2px] transition-colors"
            >
              <Icon name="plus" className="text-[11px]" />
              <span className="hidden sm:inline">New Analysis</span>
            </Link>
          </div>
        </header>

        <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 max-w-[1600px] w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
};
