import { useState, useMemo } from 'react';
import { useStore } from '../store';
import { vscode } from '../App';
import CodeLinks from './CodeLinks';
import type { LctlEvent } from '../types';

type TabId = 'overview' | 'data' | 'related';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

const TABS: Tab[] = [
  { id: 'overview', label: 'Overview', icon: '📋' },
  { id: 'data', label: 'Raw Data', icon: '{ }' },
  { id: 'related', label: 'Related', icon: '🔗' },
];

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(2)}m`;
}

function ConfidenceMeter({ value }: { value: number }) {
  const percentage = Math.round(value * 100);
  const color = value >= 0.8 ? 'var(--vscode-success)' :
                value >= 0.6 ? 'var(--vscode-warning)' :
                'var(--vscode-error)';

  return (
    <div className="confidence-meter">
      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{ width: `${percentage}%`, background: color }}
        />
      </div>
      <span className="confidence-value" style={{ color }}>{percentage}%</span>
    </div>
  );
}

function EventTypeBadge({ type }: { type: string }) {
  const config: Record<string, { icon: string; color: string; label: string }> = {
    step_start: { icon: '▶', color: 'var(--color-step-start)', label: 'Step Start' },
    step_end: { icon: '⏹', color: 'var(--color-step-end)', label: 'Step End' },
    tool_call: { icon: '🔧', color: 'var(--color-tool-call)', label: 'Tool Call' },
    fact_added: { icon: '💡', color: 'var(--color-fact)', label: 'Fact Added' },
    fact_modified: { icon: '✏', color: 'var(--color-fact)', label: 'Fact Modified' },
    error: { icon: '❌', color: 'var(--vscode-error)', label: 'Error' },
    checkpoint: { icon: '📍', color: 'var(--color-checkpoint)', label: 'Checkpoint' },
  };

  const { icon, color, label } = config[type] || { icon: '•', color: 'var(--vscode-foreground)', label: type };

  return (
    <span className="event-type-badge" style={{ '--badge-color': color } as React.CSSProperties}>
      <span className="badge-icon">{icon}</span>
      <span className="badge-label">{label}</span>
    </span>
  );
}

function OverviewTab({ event }: { event: LctlEvent }) {
  const { seq, type, timestamp, agent, data } = event;

  // Build field list based on event type
  const fields: Array<{ label: string; value: React.ReactNode; fullWidth?: boolean }> = [];

  // Common fields
  fields.push({ label: 'Sequence', value: <span className="mono">#{seq}</span> });
  fields.push({ label: 'Timestamp', value: new Date(timestamp).toLocaleString() });

  if (agent) {
    fields.push({
      label: 'Agent',
      value: <span className="agent-badge">{agent}</span>
    });
  }

  // Type-specific fields
  if (type === 'step_start' && data) {
    if (data.intent) fields.push({ label: 'Intent', value: String(data.intent) });
    if (data.input_summary) fields.push({ label: 'Input Summary', value: String(data.input_summary), fullWidth: true });
  }

  if (type === 'step_end' && data) {
    if (data.outcome) {
      const isSuccess = data.outcome === 'success';
      fields.push({
        label: 'Outcome',
        value: (
          <span className={`outcome-badge ${isSuccess ? 'success' : 'error'}`}>
            {isSuccess ? '✓' : '✗'} {String(data.outcome)}
          </span>
        )
      });
    }
    if (data.duration_ms) fields.push({ label: 'Duration', value: formatDuration(Number(data.duration_ms)) });
    // Library traces nest tokens as data.tokens.{input,output}; older ones are flat.
    const tokens = (data.tokens && typeof data.tokens === 'object' ? data.tokens : {}) as Record<string, unknown>;
    const tokensIn = Number(tokens.input ?? data.tokens_in) || 0;
    const tokensOut = Number(tokens.output ?? data.tokens_out) || 0;
    if (tokensIn) fields.push({ label: 'Tokens In', value: tokensIn.toLocaleString() });
    if (tokensOut) fields.push({ label: 'Tokens Out', value: tokensOut.toLocaleString() });
    if (data.output_summary) fields.push({ label: 'Output Summary', value: String(data.output_summary), fullWidth: true });
  }

  if (type === 'tool_call' && data) {
    if (data.tool) fields.push({ label: 'Tool', value: <span className="tool-badge">{String(data.tool)}</span> });
    if (data.duration_ms) fields.push({ label: 'Duration', value: formatDuration(Number(data.duration_ms)) });
  }

  if ((type === 'fact_added' || type === 'fact_modified') && data) {
    if (data.id) fields.push({ label: 'Fact ID', value: <span className="mono">{String(data.id)}</span> });
    if (data.confidence !== undefined) {
      fields.push({
        label: 'Confidence',
        value: <ConfidenceMeter value={Number(data.confidence)} />
      });
    }
    if (data.text) fields.push({ label: 'Fact Text', value: String(data.text), fullWidth: true });
    if (data.reason) fields.push({ label: 'Reason', value: String(data.reason), fullWidth: true });
  }

  if (type === 'error' && data) {
    if (data.error_type) fields.push({ label: 'Error Type', value: <span className="error-type">{String(data.error_type)}</span> });
    if (data.category) fields.push({ label: 'Category', value: String(data.category) });
    if (data.message) fields.push({ label: 'Message', value: String(data.message), fullWidth: true });
    if (data.recoverable !== undefined) {
      fields.push({
        label: 'Recoverable',
        value: data.recoverable ? <span className="success">Yes</span> : <span className="error">No</span>
      });
    }
  }

  return (
    <div className="overview-tab">
      <div className="event-header-enhanced">
        <EventTypeBadge type={type} />
        <span className="event-seq">Event #{seq}</span>
      </div>

      <div className="fields-grid">
        {fields.map((field, i) => (
          <div
            key={`${field.label}-${i}`}
            className={`field-item ${field.fullWidth ? 'full-width' : ''}`}
          >
            <div className="field-label">{field.label}</div>
            <div className="field-value">{field.value}</div>
          </div>
        ))}
      </div>

      <CodeLinks event={event} />
    </div>
  );
}

function DataTab({ event }: { event: LctlEvent }) {
  const [copied, setCopied] = useState(false);
  const jsonString = JSON.stringify(event.data || {}, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = jsonString;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!event.data || Object.keys(event.data).length === 0) {
    return (
      <div className="empty-tab">
        <span className="empty-icon">📭</span>
        <span>No data payload for this event</span>
      </div>
    );
  }

  return (
    <div className="data-tab">
      <div className="data-header">
        <span className="data-title">Event Data Payload</span>
        <button className="copy-button" onClick={handleCopy}>
          {copied ? '✓ Copied!' : '📋 Copy'}
        </button>
      </div>
      <pre className="json-viewer">{jsonString}</pre>
    </div>
  );
}

function RelatedTab({ event }: { event: LctlEvent }) {
  const chain = useStore((s) => s.chain);
  const selectEvent = useStore((s) => s.selectEvent);

  const relatedEvents = useMemo(() => {
    if (!chain?.events) return { sameAgent: [], sameType: [], nearby: [] };

    const events = chain.events;

    // Same agent events (before and after)
    const sameAgent = events
      .filter(e => e.agent === event.agent && e.seq !== event.seq)
      .slice(0, 5);

    // Same type events
    const sameType = events
      .filter(e => e.type === event.type && e.seq !== event.seq)
      .slice(0, 5);

    // Nearby events (within 5 sequences)
    const nearby = events
      .filter(e => Math.abs(e.seq - event.seq) <= 5 && e.seq !== event.seq)
      .sort((a, b) => a.seq - b.seq);

    return { sameAgent, sameType, nearby };
  }, [chain, event]);

  const renderEventLink = (e: LctlEvent) => (
    <button
      key={e.seq}
      className="related-event-link"
      onClick={() => selectEvent(e)}
    >
      <span className="related-seq">#{e.seq}</span>
      <span className="related-type">{e.type}</span>
      {e.agent && <span className="related-agent">{e.agent}</span>}
    </button>
  );

  return (
    <div className="related-tab">
      {event.agent && relatedEvents.sameAgent.length > 0 && (
        <div className="related-section">
          <div className="related-section-title">Same Agent ({event.agent})</div>
          <div className="related-list">{relatedEvents.sameAgent.map(renderEventLink)}</div>
        </div>
      )}

      {relatedEvents.sameType.length > 0 && (
        <div className="related-section">
          <div className="related-section-title">Same Type ({event.type})</div>
          <div className="related-list">{relatedEvents.sameType.map(renderEventLink)}</div>
        </div>
      )}

      {relatedEvents.nearby.length > 0 && (
        <div className="related-section">
          <div className="related-section-title">Nearby Events</div>
          <div className="related-list">{relatedEvents.nearby.map(renderEventLink)}</div>
        </div>
      )}
    </div>
  );
}

export default function EventDetails() {
  const selectedEvent = useStore((s) => s.selectedEvent);
  const selectEvent = useStore((s) => s.selectEvent);
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  if (!selectedEvent) {
    return (
      <div className="details-panel-enhanced">
        <div className="details-header-enhanced">
          <span className="header-title">Event Details</span>
        </div>
        <div className="empty-details">
          <span className="empty-icon-large">🔍</span>
          <span className="empty-text">Select an event to view details</span>
          <span className="empty-hint">Click on any event in the timeline</span>
        </div>
      </div>
    );
  }

  // Check for file paths in data to enable navigation
  const filePath = (selectedEvent.data?.file_path || selectedEvent.data?.path || selectedEvent.data?.source_file) as string | undefined;
  const lineNumber = (selectedEvent.data?.line || selectedEvent.data?.line_number) as number | undefined;

  const handleOpenFile = () => {
    if (filePath) {
      vscode.postMessage({
        type: 'openFile',
        path: String(filePath),
        line: lineNumber ? Number(lineNumber) : undefined
      });
    }
  };

  return (
    <div className="details-panel-enhanced">
      <div className="details-header-enhanced">
        <span className="header-title">Event #{selectedEvent.seq}</span>
        <div className="header-actions">
          {filePath && (
            <button className="action-button" onClick={handleOpenFile} title="Open source file">
              📄
            </button>
          )}
          <button
            className="action-button close"
            onClick={() => selectEvent(null)}
            title="Close details"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="details-tabs">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="details-content-enhanced">
        {activeTab === 'overview' && <OverviewTab event={selectedEvent} />}
        {activeTab === 'data' && <DataTab event={selectedEvent} />}
        {activeTab === 'related' && <RelatedTab event={selectedEvent} />}
      </div>
    </div>
  );
}
