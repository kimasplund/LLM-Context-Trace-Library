// LCTL v4.0 Types
export interface LctlChainFile {
  lctl: string;
  chain: {
    id: string;
  };
  events: LctlEvent[];
}

export interface LctlEvent {
  seq: number;
  type: EventType;
  timestamp: string;
  agent?: string;
  data?: Record<string, unknown>;
}

export type EventType =
  | 'step_start'
  | 'step_end'
  | 'tool_call'
  | 'fact_added'
  | 'fact_modified'
  | 'error'
  | 'checkpoint';

// UI State Types
export interface ChainStats {
  eventCount: number;
  errorCount: number;
  warningCount: number;
  totalDurationMs: number;
  tokensIn: number;
  tokensOut: number;
  estimatedCost: number;
  agents: string[];
  tools: string[];
  facts: FactInfo[];
}

export interface FactInfo {
  id: string;
  text: string;
  confidence: number;
  source?: string;
  lastModified?: string;
}

export interface TimelineTrack {
  agent: string;
  events: LctlEvent[];
  startTime: number;
  endTime: number;
}

export interface ReplayState {
  isPlaying: boolean;
  currentSeq: number;
  speed: number;
  maxSeq: number;
}

export interface FilterState {
  eventTypes: EventType[];
  agents: string[];
  searchQuery: string;
  timeRange: [number, number] | null;
}

// VS Code API Types
declare global {
  function acquireVsCodeApi(): VsCodeApi;
}

export interface VsCodeApi {
  postMessage(message: WebviewMessage): void;
  getState<T>(): T | undefined;
  setState<T>(state: T): void;
}

// Message Types
export type WebviewMessage =
  | { type: 'ready' }
  | { type: 'loadChain'; path: string }
  | { type: 'replay'; targetSeq: number }
  | { type: 'openFile'; path: string; line?: number }
  | { type: 'exportHtml' }
  | { type: 'showStats' }
  | { type: 'requestData' }
  | { type: 'requestCompare' };

export type ExtensionMessage =
  | { type: 'chainData'; data: LctlChainFile; path: string; isRecording: boolean; pricing?: { input: number; output: number } }
  | { type: 'compareChainData'; data: LctlChainFile; path: string }
  | { type: 'error'; message: string }
  | { type: 'recordingUpdate'; isRecording: boolean };
