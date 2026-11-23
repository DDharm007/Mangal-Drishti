import React, { useCallback, useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DatasetResearchPage } from "./pages/DatasetResearchPage";
import { ModelPerformancePage } from "./pages/ModelPerformancePage";
import { OverviewPage } from "./pages/OverviewPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SystemStatusPage } from "./pages/SystemStatusPage";
import { TerrainAnalysisPage } from "./pages/TerrainAnalysisPage";
import { fetchSystemStatus } from "./services/api";
import type { SystemStatusResponse } from "./types/api";

export function App(): React.ReactElement {
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<string>("deeplabv3plus");

  const refreshStatus = useCallback(async () => {
    try {
      const s = await fetchSystemStatus();
      setSystemStatus(s);
      setStatusError(null);
    } catch (e) {
      setStatusError(e instanceof Error ? e.message : "Backend API unreachable");
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  return (
    <BrowserRouter>
      <AppShell
        systemStatus={systemStatus}
        statusError={statusError}
        activeModel={activeModel}
      >
        <Routes>
          <Route path="/" element={<OverviewPage systemStatus={systemStatus} statusError={statusError} />} />
          <Route
            path="/analyze"
            element={
              <TerrainAnalysisPage
                systemStatus={systemStatus}
                activeModel={activeModel}
                onModelChange={(m) => {
                  setActiveModel(m);
                  void refreshStatus();
                }}
                onAnalysisComplete={() => void refreshStatus()}
              />
            }
          />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/models" element={<ModelPerformancePage />} />
          <Route path="/research" element={<DatasetResearchPage />} />
          <Route
            path="/status"
            element={
              <SystemStatusPage
                systemStatus={systemStatus}
                statusError={statusError}
                onRefresh={() => void refreshStatus()}
              />
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
