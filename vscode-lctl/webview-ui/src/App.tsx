import { useEffect, useCallback, useRef, useState } from 'react';
import { useStore } from './store';
import { getVsCodeApi } from './hooks/useVsCode';
import Header from './components/Header';
import StatsGrid from './components/StatsGrid';
import Timeline from './components/Timeline';
import FilterPanel from './components/FilterPanel';
import EventDetails from './components/EventDetails';
import ReplayBar from './components/ReplayBar';
import DiffPanel from './components/DiffPanel';
import Breadcrumb from './components/Breadcrumb';
import Minimap from './components/Minimap';
import { KeyboardShortcutsOverlay, useKeyboardShortcuts } from './components/KeyboardShortcuts';
import type { ExtensionMessage } from './types';

// Get VS Code API (cached)
export const vscode = getVsCodeApi();

function ResizeHandle({ onResize }: { onResize: (delta: number) => void }) {
  const [isDragging, setIsDragging] = useState(false);
  const startY = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startY.current = e.clientY;
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = startY.current - e.clientY;
      startY.current = e.clientY;
      onResize(delta);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, onResize]);

  return (
    <div
      className={`resize-handle ${isDragging ? 'active' : ''}`}
      onMouseDown={handleMouseDown}
    >
      <div className="resize-handle-bar" />
    </div>
  );
}

function App() {
  const chain = useStore((s) => s.chain);
  const setChain = useStore((s) => s.setChain);
  const setCompareChain = useStore((s) => s.setCompareChain);
  const compareChain = useStore((s) => s.compareChain);
  const isDiffMode = useStore((s) => s.isDiffMode);
  const toggleDiffMode = useStore((s) => s.toggleDiffMode);
  const detailsPanelHeight = useStore((s) => s.detailsPanelHeight);
  const setDetailsPanelHeight = useStore((s) => s.setDetailsPanelHeight);

  // Keyboard shortcuts hook
  const { showShortcuts, setShowShortcuts } = useKeyboardShortcuts();

  useEffect(() => {
    // Listen for messages from extension
    const handleMessage = (event: MessageEvent<ExtensionMessage>) => {
      const message = event.data;

      switch (message.type) {
        case 'chainData':
          setChain(message.data, message.path, message.isRecording, message.pricing);
          break;
        case 'compareChainData':
          setCompareChain(message.data, message.path);
          useStore.setState({ isDiffMode: true });
          break;
        case 'recordingUpdate':
          useStore.setState({ isRecording: message.isRecording });
          break;
        case 'error':
          console.error('Extension error:', message.message);
          break;
      }
    };

    window.addEventListener('message', handleMessage);

    // Request initial data
    vscode.postMessage({ type: 'ready' });

    return () => window.removeEventListener('message', handleMessage);
  }, [setChain, setCompareChain]);

  const handlePanelResize = useCallback((delta: number) => {
    setDetailsPanelHeight(detailsPanelHeight + delta);
  }, [detailsPanelHeight, setDetailsPanelHeight]);

  if (!chain) {
    return (
      <div className="app-container">
        <Header />
        <div className="empty-state">
          <div className="empty-state-icon">🔍</div>
          <div>No chain loaded</div>
          <div style={{ fontSize: 12, opacity: 0.6 }}>
            Select a chain file from the LCTL Explorer or load one manually
          </div>
        </div>
      </div>
    );
  }

  // Show diff view if in diff mode
  if (isDiffMode) {
    return (
      <div className="app-container">
        <Header />
        <DiffPanel
          chain1={chain}
          chain2={compareChain}
          onClose={() => {
            setCompareChain(null, null);
            toggleDiffMode();
          }}
        />
      </div>
    );
  }

  return (
    <div className="app-container">
      <Header />
      <StatsGrid />
      <Minimap />
      <Breadcrumb />
      <div className="main-content">
        <div className="sidebar">
          <FilterPanel />
        </div>
        <div className="content-area">
          <Timeline />
          <ResizeHandle onResize={handlePanelResize} />
          <div style={{ height: detailsPanelHeight, flexShrink: 0 }}>
            <EventDetails />
          </div>
        </div>
      </div>
      <ReplayBar />
      <KeyboardShortcutsOverlay
        isOpen={showShortcuts}
        onClose={() => setShowShortcuts(false)}
      />
    </div>
  );
}

export default App;
