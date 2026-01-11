/**
 * LCTL Dashboard - Interactive visualization for multi-agent workflows
 */

// State
const state = {
    chains: [],
    currentChain: null,
    currentSeq: null,
    maxSeq: 0,
    selectedEvent: null,
    isPlaying: false,
    playInterval: null,
    eventFilter: 'all',
    websocket: null,
    wsConnected: false,
    wsClientId: null,
    liveMode: false,
    liveEvents: []
};

// Agent colors for consistent coloring
const agentColors = [
    '#58a6ff', '#3fb950', '#a371f7', '#d29922', '#39c5cf', '#db6d28'
];

const agentColorMap = new Map();

function getAgentColor(agent) {
    if (!agentColorMap.has(agent)) {
        const colorIndex = agentColorMap.size % agentColors.length;
        agentColorMap.set(agent, agentColors[colorIndex]);
    }
    return agentColorMap.get(agent);
}

// DOM Elements
const elements = {
    chainSelector: document.getElementById('chain-selector'),
    refreshBtn: document.getElementById('refresh-btn'),
    emptyState: document.getElementById('empty-state'),
    dashboard: document.getElementById('dashboard'),
    workingDirPath: document.getElementById('working-dir-path'),
    timeSlider: document.getElementById('time-slider'),
    currentSeq: document.getElementById('current-seq'),
    maxSeq: document.getElementById('max-seq'),
    playBtn: document.getElementById('play-btn'),
    playIcon: document.getElementById('play-icon'),
    pauseIcon: document.getElementById('pause-icon'),
    stepBackBtn: document.getElementById('step-back-btn'),
    stepForwardBtn: document.getElementById('step-forward-btn'),
    timeline: document.getElementById('timeline'),
    swimlanes: document.getElementById('swimlanes'),
    factRegistry: document.getElementById('fact-registry'),
    eventDetails: document.getElementById('event-details'),
    selectedEventLabel: document.getElementById('selected-event-label'),
    bottlenecks: document.getElementById('bottlenecks'),
    eventFilter: document.getElementById('event-filter'),
    toastContainer: document.getElementById('toast-container'),
    // Stats
    statEvents: document.getElementById('stat-events'),
    statAgents: document.getElementById('stat-agents'),
    statFacts: document.getElementById('stat-facts'),
    statDuration: document.getElementById('stat-duration'),
    statTokens: document.getElementById('stat-tokens'),
    statErrors: document.getElementById('stat-errors')
};

// Initialize
document.addEventListener('DOMContentLoaded', init);

async function init() {
    await loadChains();
    setupEventListeners();
}

function setupEventListeners() {
    elements.chainSelector.addEventListener('change', onChainSelect);
    elements.refreshBtn.addEventListener('click', loadChains);
    elements.timeSlider.addEventListener('input', onSliderChange);
    elements.playBtn.addEventListener('click', togglePlay);
    elements.stepBackBtn.addEventListener('click', stepBack);
    elements.stepForwardBtn.addEventListener('click', stepForward);
    elements.eventFilter.addEventListener('change', onFilterChange);
}

// API Functions
async function loadChains() {
    try {
        const response = await fetch('/api/chains');
        const data = await response.json();

        state.chains = data.chains;
        elements.workingDirPath.textContent = data.working_dir;

        // Update selector
        elements.chainSelector.innerHTML = '<option value="">Select a chain...</option>';
        for (const chain of state.chains) {
            const option = document.createElement('option');
            option.value = chain.filename;
            option.textContent = `${chain.id} (${chain.event_count} events)`;
            if (chain.error) {
                option.textContent += ' [Error]';
                option.disabled = true;
            }
            elements.chainSelector.appendChild(option);
        }

        showToast('Chains loaded', 'success');
    } catch (error) {
        console.error('Failed to load chains:', error);
        showToast('Failed to load chains', 'error');
    }
}

async function loadChain(filename) {
    try {
        const response = await fetch(`/api/chain/${encodeURIComponent(filename)}`);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const data = await response.json();

        state.currentChain = data;
        state.maxSeq = data.events.length > 0 ? data.events[data.events.length - 1].seq : 0;
        state.currentSeq = state.maxSeq;
        state.selectedEvent = null;

        // Reset agent colors
        agentColorMap.clear();

        // Update UI
        showDashboard();
        updateStats();
        updateSlider();
        renderTimeline();
        renderSwimlanes();
        renderFactRegistry();
        renderBottlenecks();
        clearEventDetails();

        showToast(`Loaded chain: ${data.chain.id}`, 'success');
    } catch (error) {
        console.error('Failed to load chain:', error);
        showToast('Failed to load chain', 'error');
    }
}

async function replayTo(targetSeq) {
    if (!state.currentChain) return;

    try {
        const response = await fetch('/api/replay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: state.currentChain.chain.filename,
                target_seq: targetSeq
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        state.currentSeq = targetSeq;

        // Update visuals to reflect current state
        updateStats(data.state);
        renderFactRegistry(data.state.facts);
        updateTimelineVisibility();
        updateSwimlanesVisibility();

    } catch (error) {
        console.error('Failed to replay:', error);
        showToast('Failed to replay', 'error');
    }
}

// Event Handlers
async function onChainSelect(e) {
    const filename = e.target.value;
    if (filename) {
        await loadChain(filename);
    } else {
        hideDashboard();
    }
}

function onSliderChange(e) {
    const seq = parseInt(e.target.value);
    state.currentSeq = seq;
    elements.currentSeq.textContent = `Seq: ${seq}`;
    replayTo(seq);
}

function onFilterChange(e) {
    state.eventFilter = e.target.value;
    renderTimeline();
}

function togglePlay() {
    state.isPlaying = !state.isPlaying;

    if (state.isPlaying) {
        elements.playIcon.classList.add('hidden');
        elements.pauseIcon.classList.remove('hidden');

        // Reset to beginning if at end
        if (state.currentSeq >= state.maxSeq) {
            state.currentSeq = 1;
            elements.timeSlider.value = 1;
        }

        state.playInterval = setInterval(() => {
            if (state.currentSeq < state.maxSeq) {
                state.currentSeq++;
                elements.timeSlider.value = state.currentSeq;
                elements.currentSeq.textContent = `Seq: ${state.currentSeq}`;
                replayTo(state.currentSeq);
            } else {
                togglePlay(); // Stop at end
            }
        }, 500);
    } else {
        elements.playIcon.classList.remove('hidden');
        elements.pauseIcon.classList.add('hidden');
        if (state.playInterval) {
            clearInterval(state.playInterval);
            state.playInterval = null;
        }
    }
}

function stepBack() {
    if (state.currentSeq > 1) {
        state.currentSeq--;
        elements.timeSlider.value = state.currentSeq;
        elements.currentSeq.textContent = `Seq: ${state.currentSeq}`;
        replayTo(state.currentSeq);
    }
}

function stepForward() {
    if (state.currentSeq < state.maxSeq) {
        state.currentSeq++;
        elements.timeSlider.value = state.currentSeq;
        elements.currentSeq.textContent = `Seq: ${state.currentSeq}`;
        replayTo(state.currentSeq);
    }
}

// UI Functions
function showDashboard() {
    elements.emptyState.classList.add('hidden');
    elements.dashboard.classList.remove('hidden');
}

function hideDashboard() {
    elements.emptyState.classList.remove('hidden');
    elements.dashboard.classList.add('hidden');
    state.currentChain = null;
}

function updateSlider() {
    elements.timeSlider.min = 1;
    elements.timeSlider.max = state.maxSeq;
    elements.timeSlider.value = state.currentSeq;
    elements.currentSeq.textContent = `Seq: ${state.currentSeq}`;
    elements.maxSeq.textContent = `Max: ${state.maxSeq}`;
}

function updateStats(replayState = null) {
    if (!state.currentChain) return;

    const chain = state.currentChain;
    const stateData = replayState || chain.state;

    elements.statEvents.textContent = chain.events.length;
    elements.statAgents.textContent = chain.agents.length;
    elements.statFacts.textContent = Object.keys(stateData.facts).length;
    elements.statDuration.textContent = formatDuration(stateData.metrics.total_duration_ms);
    elements.statTokens.textContent = formatNumber(
        stateData.metrics.total_tokens_in + stateData.metrics.total_tokens_out
    );
    elements.statErrors.textContent = stateData.metrics.error_count;
}

function renderTimeline() {
    if (!state.currentChain) return;

    const container = elements.timeline;
    container.innerHTML = '';

    const events = state.currentChain.events.filter(e => {
        if (state.eventFilter === 'all') return true;
        return e.type === state.eventFilter;
    });

    for (const event of events) {
        const el = document.createElement('div');
        el.className = 'timeline-event';
        if (event.seq > state.currentSeq) {
            el.classList.add('dimmed');
        }
        if (state.selectedEvent && state.selectedEvent.seq === event.seq) {
            el.classList.add('selected');
        }

        el.innerHTML = `
            <span class="event-seq">#${event.seq}</span>
            <div class="event-info">
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="event-type-badge ${event.type}">${formatEventType(event.type)}</span>
                    <span class="event-agent" style="color: ${getAgentColor(event.agent)}">${event.agent}</span>
                </div>
                <div class="event-summary">${getEventSummary(event)}</div>
            </div>
        `;

        el.addEventListener('click', () => selectEvent(event));
        container.appendChild(el);
    }
}

function renderSwimlanes() {
    if (!state.currentChain) return;

    const container = elements.swimlanes;
    container.innerHTML = '';

    const chain = state.currentChain;
    const bottleneckSeqs = new Set(
        chain.analysis.bottlenecks.slice(0, 3).map(b => b.seq)
    );

    for (const agent of chain.agents) {
        const agentEvents = chain.events.filter(e => e.agent === agent);
        if (agentEvents.length === 0) continue;

        const lane = document.createElement('div');
        lane.className = 'swimlane';

        const header = document.createElement('div');
        header.className = 'swimlane-header';
        header.innerHTML = `
            <div class="agent-indicator" style="background-color: ${getAgentColor(agent)}"></div>
            <span class="agent-name">${agent}</span>
            <span style="color: var(--text-muted); font-size: 0.75rem;">(${agentEvents.length} events)</span>
        `;
        lane.appendChild(header);

        const eventsContainer = document.createElement('div');
        eventsContainer.className = 'swimlane-events';

        for (const event of agentEvents) {
            const el = document.createElement('div');
            el.className = 'swimlane-event';

            if (event.seq > state.currentSeq) {
                el.classList.add('dimmed');
            }
            if (state.selectedEvent && state.selectedEvent.seq === event.seq) {
                el.classList.add('selected');
            }
            if (bottleneckSeqs.has(event.seq)) {
                el.classList.add('bottleneck');
            }
            if (event.type === 'error') {
                el.classList.add('error');
            }

            el.innerHTML = `
                <span class="event-type-badge ${event.type}">${formatEventType(event.type)}</span>
                <span>#${event.seq}</span>
            `;

            el.addEventListener('click', () => selectEvent(event));
            eventsContainer.appendChild(el);
        }

        lane.appendChild(eventsContainer);
        container.appendChild(lane);
    }
}

function renderFactRegistry(facts = null) {
    if (!state.currentChain) return;

    const container = elements.factRegistry;
    container.innerHTML = '';

    const factsData = facts || state.currentChain.state.facts;

    if (Object.keys(factsData).length === 0) {
        container.innerHTML = '<div class="no-selection">No facts at this point in time</div>';
        return;
    }

    for (const [factId, fact] of Object.entries(factsData)) {
        const el = document.createElement('div');
        el.className = 'fact-item';

        const confidence = fact.confidence || 1.0;
        const confidenceClass = confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';

        el.innerHTML = `
            <div class="fact-header">
                <span class="fact-id">${factId}</span>
                <div class="fact-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" style="width: ${confidence * 100}%"></div>
                    </div>
                    <span class="confidence-value">${(confidence * 100).toFixed(0)}%</span>
                </div>
            </div>
            <div class="fact-text">${escapeHtml(fact.text || '')}</div>
            <div class="fact-source">Source: ${fact.source || 'unknown'}</div>
        `;

        container.appendChild(el);
    }
}

function renderBottlenecks() {
    if (!state.currentChain) return;

    const container = elements.bottlenecks;
    container.innerHTML = '';

    const bottlenecks = state.currentChain.analysis.bottlenecks.slice(0, 5);

    if (bottlenecks.length === 0) {
        container.innerHTML = '<div class="no-selection">No bottleneck data available</div>';
        return;
    }

    for (let i = 0; i < bottlenecks.length; i++) {
        const b = bottlenecks[i];
        const el = document.createElement('div');
        el.className = 'bottleneck-item';

        el.innerHTML = `
            <span class="bottleneck-rank">#${i + 1}</span>
            <div class="bottleneck-info">
                <div class="bottleneck-agent" style="color: ${getAgentColor(b.agent)}">${b.agent}</div>
                <div class="bottleneck-details">Seq ${b.seq} - ${formatDuration(b.duration_ms)}</div>
            </div>
            <div class="bottleneck-bar">
                <div class="bottleneck-fill" style="width: ${Math.min(b.percentage, 100)}%"></div>
            </div>
            <span class="bottleneck-percentage">${b.percentage.toFixed(0)}%</span>
        `;

        container.appendChild(el);
    }
}

function selectEvent(event) {
    state.selectedEvent = event;

    // Update timeline
    document.querySelectorAll('.timeline-event').forEach(el => {
        el.classList.remove('selected');
    });
    document.querySelectorAll('.swimlane-event').forEach(el => {
        el.classList.remove('selected');
    });

    // Re-render to show selection
    renderTimeline();
    renderSwimlanes();
    showEventDetails(event);
}

function showEventDetails(event) {
    const container = elements.eventDetails;
    elements.selectedEventLabel.textContent = `Event #${event.seq}`;

    let detailsHtml = `
        <div class="detail-grid">
            <div class="detail-item">
                <span class="detail-label">Sequence</span>
                <span class="detail-value">${event.seq}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Type</span>
                <span class="detail-value"><span class="event-type-badge ${event.type}">${formatEventType(event.type)}</span></span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Agent</span>
                <span class="detail-value" style="color: ${getAgentColor(event.agent)}">${event.agent}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Timestamp</span>
                <span class="detail-value">${formatTimestamp(event.timestamp)}</span>
            </div>
        </div>
    `;

    // Add data details
    if (event.data && Object.keys(event.data).length > 0) {
        detailsHtml += `
            <div style="margin-top: var(--spacing-md);">
                <span class="detail-label">Data</span>
                <pre class="detail-value code">${escapeHtml(JSON.stringify(event.data, null, 2))}</pre>
            </div>
        `;
    }

    container.innerHTML = detailsHtml;
}

function clearEventDetails() {
    elements.eventDetails.innerHTML = '<div class="no-selection">Click on any event in the timeline or swim lanes to view details</div>';
    elements.selectedEventLabel.textContent = 'Select an event to view details';
}

function updateTimelineVisibility() {
    document.querySelectorAll('.timeline-event').forEach(el => {
        const seq = parseInt(el.querySelector('.event-seq').textContent.slice(1));
        if (seq > state.currentSeq) {
            el.classList.add('dimmed');
        } else {
            el.classList.remove('dimmed');
        }
    });
}

function updateSwimlanesVisibility() {
    document.querySelectorAll('.swimlane-event').forEach(el => {
        const seqText = el.textContent.match(/#(\d+)/);
        if (seqText) {
            const seq = parseInt(seqText[1]);
            if (seq > state.currentSeq) {
                el.classList.add('dimmed');
            } else {
                el.classList.remove('dimmed');
            }
        }
    });
}

// Utility Functions
function formatEventType(type) {
    const typeMap = {
        'step_start': 'Start',
        'step_end': 'End',
        'fact_added': 'Fact+',
        'fact_modified': 'Fact~',
        'tool_call': 'Tool',
        'error': 'Error',
        'checkpoint': 'Chkpt',
        'stream_start': 'Stream',
        'stream_chunk': 'Chunk',
        'stream_end': 'StreamEnd',
        'contract_validation': 'Contract',
        'model_routing': 'Route'
    };
    return typeMap[type] || type;
}

function getEventSummary(event) {
    switch (event.type) {
        case 'step_start':
            return event.data.intent || event.data.input_summary || '';
        case 'step_end':
            return event.data.output_summary || event.data.outcome || '';
        case 'fact_added':
        case 'fact_modified':
            return `${event.data.id}: ${(event.data.text || '').slice(0, 50)}`;
        case 'tool_call':
            return `${event.data.tool} (${event.data.duration_ms || 0}ms)`;
        case 'error':
            return event.data.message || event.data.type || '';
        case 'checkpoint':
            return `Hash: ${event.data.state_hash || 'unknown'}`;
        default:
            return '';
    }
}

function formatDuration(ms) {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
}

function formatNumber(n) {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
}

function formatTimestamp(ts) {
    try {
        const date = new Date(ts);
        return date.toLocaleTimeString();
    } catch {
        return ts;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================================================
// EVALUATION FEATURES
// ============================================================================

// Additional elements for evaluation features
const evalElements = {
    tabs: null,
    compareBtn: document.getElementById('compare-btn'),
    compareModal: document.getElementById('compare-modal'),
    closeModal: document.getElementById('close-modal'),
    compareChain1: document.getElementById('compare-chain1'),
    compareChain2: document.getElementById('compare-chain2'),
    runCompare: document.getElementById('run-compare'),
    compareResults: document.getElementById('compare-results'),
    chain1Summary: document.getElementById('chain1-summary'),
    chain2Summary: document.getElementById('chain2-summary'),
    divergenceInfo: document.getElementById('divergence-info'),
    diffList: document.getElementById('diff-list'),
    factDiff: document.getElementById('fact-diff'),
    latencyChart: document.getElementById('latency-chart'),
    tokenChart: document.getElementById('token-chart'),
    agentMetricsTable: document.getElementById('agent-metrics-table'),
    errorTimeline: document.getElementById('error-timeline'),
    overallScore: document.getElementById('overall-score'),
    perfScore: document.getElementById('perf-score'),
    reliabilityScore: document.getElementById('reliability-score'),
    qualityScore: document.getElementById('quality-score'),
    overallRing: document.getElementById('overall-ring'),
    perfBar: document.getElementById('perf-bar'),
    reliabilityBar: document.getElementById('reliability-bar'),
    qualityBar: document.getElementById('quality-bar'),
    issuesList: document.getElementById('issues-list'),
    confidenceHeatmap: document.getElementById('confidence-heatmap'),
    evalSummary: document.getElementById('eval-summary')
};

// Evaluation state
const evalState = {
    metricsData: null,
    evaluationData: null,
    latencyChartInstance: null,
    tokenChartInstance: null
};

// Initialize evaluation features after DOM is ready
function initEvalFeatures() {
    evalElements.tabs = document.querySelectorAll('.tab');

    setupTabNavigation();
    setupCompareModal();

    if (evalElements.compareBtn) {
        evalElements.compareBtn.addEventListener('click', openCompareModal);
    }
}

// Call init after main init
const originalInit = init;
init = async function() {
    await originalInit();
    initEvalFeatures();
};

// Tab Navigation
function setupTabNavigation() {
    if (!evalElements.tabs) return;

    evalElements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            evalElements.tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }

            if (targetId === 'metrics-tab' && state.currentChain) {
                loadMetrics();
            } else if (targetId === 'evaluation-tab' && state.currentChain) {
                loadEvaluation();
            }
        });
    });
}

// Comparison Modal
function setupCompareModal() {
    if (evalElements.closeModal) {
        evalElements.closeModal.addEventListener('click', closeCompareModal);
    }

    if (evalElements.compareModal) {
        const backdrop = evalElements.compareModal.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.addEventListener('click', closeCompareModal);
        }
    }

    if (evalElements.compareChain1) {
        evalElements.compareChain1.addEventListener('change', updateCompareButton);
    }
    if (evalElements.compareChain2) {
        evalElements.compareChain2.addEventListener('change', updateCompareButton);
    }
    if (evalElements.runCompare) {
        evalElements.runCompare.addEventListener('click', runComparison);
    }
}

function openCompareModal() {
    if (!evalElements.compareModal) return;

    populateCompareSelectors();
    evalElements.compareModal.classList.remove('hidden');
    evalElements.compareResults.classList.add('hidden');
}

function closeCompareModal() {
    if (evalElements.compareModal) {
        evalElements.compareModal.classList.add('hidden');
    }
}

function populateCompareSelectors() {
    const chains = state.chains.filter(c => !c.error);

    [evalElements.compareChain1, evalElements.compareChain2].forEach(select => {
        if (!select) return;
        select.innerHTML = '<option value="">Select a chain...</option>';
        chains.forEach(chain => {
            const option = document.createElement('option');
            option.value = chain.filename;
            option.textContent = `${chain.id} (${chain.event_count} events)`;
            select.appendChild(option);
        });
    });
}

function updateCompareButton() {
    if (!evalElements.runCompare) return;

    const chain1 = evalElements.compareChain1?.value;
    const chain2 = evalElements.compareChain2?.value;

    evalElements.runCompare.disabled = !chain1 || !chain2 || chain1 === chain2;
}

async function runComparison() {
    const filename1 = evalElements.compareChain1?.value;
    const filename2 = evalElements.compareChain2?.value;

    if (!filename1 || !filename2) return;

    try {
        evalElements.runCompare.disabled = true;
        evalElements.runCompare.textContent = 'Comparing...';

        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename1, filename2 })
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        renderComparisonResults(data);

        evalElements.compareResults.classList.remove('hidden');
        showToast('Comparison complete', 'success');

    } catch (error) {
        console.error('Comparison failed:', error);
        showToast('Comparison failed', 'error');
    } finally {
        evalElements.runCompare.disabled = false;
        evalElements.runCompare.textContent = 'Compare Chains';
        updateCompareButton();
    }
}

function renderComparisonResults(data) {
    renderChainSummary(evalElements.chain1Summary, data.chain1, 'Chain 1');
    renderChainSummary(evalElements.chain2Summary, data.chain2, 'Chain 2');

    if (evalElements.divergenceInfo) {
        if (data.divergence_point) {
            evalElements.divergenceInfo.innerHTML = `
                <div class="divergence-alert">
                    <span class="divergence-icon">!</span>
                    Chains diverge at sequence <strong>#${data.divergence_point}</strong>
                    <span class="diff-count">${data.diff_count} difference(s) found</span>
                </div>
            `;
        } else {
            evalElements.divergenceInfo.innerHTML = `
                <div class="divergence-success">
                    <span class="success-icon">&#10003;</span>
                    Chains are identical
                </div>
            `;
        }
    }

    if (evalElements.diffList) {
        if (data.event_diffs.length > 0) {
            evalElements.diffList.innerHTML = data.event_diffs.slice(0, 10).map(diff => `
                <div class="diff-item diff-${diff.type}">
                    <span class="diff-seq">#${diff.seq}</span>
                    <span class="diff-type">${formatDiffType(diff.type)}</span>
                    ${diff.type === 'diverged' ? `
                        <div class="diff-details">
                            <div class="diff-first">Chain 1: ${diff.first?.type || 'N/A'} - ${diff.first?.agent || 'N/A'}</div>
                            <div class="diff-second">Chain 2: ${diff.second?.type || 'N/A'} - ${diff.second?.agent || 'N/A'}</div>
                        </div>
                    ` : ''}
                </div>
            `).join('');

            if (data.event_diffs.length > 10) {
                evalElements.diffList.innerHTML += `
                    <div class="diff-more">... and ${data.event_diffs.length - 10} more differences</div>
                `;
            }
        } else {
            evalElements.diffList.innerHTML = '<div class="no-diffs">No event differences</div>';
        }
    }

    if (evalElements.factDiff) {
        const fc = data.fact_comparison;
        evalElements.factDiff.innerHTML = `
            <div class="fact-diff-summary">
                <div class="fact-diff-stat">
                    <span class="stat-num">${fc.same.length}</span>
                    <span class="stat-label">Same</span>
                </div>
                <div class="fact-diff-stat different">
                    <span class="stat-num">${fc.different.length}</span>
                    <span class="stat-label">Different</span>
                </div>
                <div class="fact-diff-stat only-first">
                    <span class="stat-num">${fc.only_in_first.length}</span>
                    <span class="stat-label">Only in Chain 1</span>
                </div>
                <div class="fact-diff-stat only-second">
                    <span class="stat-num">${fc.only_in_second.length}</span>
                    <span class="stat-label">Only in Chain 2</span>
                </div>
            </div>
            ${fc.different.length > 0 ? `
                <div class="fact-diff-details">
                    <h5>Different Facts:</h5>
                    ${fc.different.slice(0, 5).map(f => `
                        <div class="fact-diff-item">
                            <span class="fact-id">${f.id}</span>
                            <div class="fact-comparison">
                                <div class="fact-v1">C1: ${escapeHtml((f.first?.text || '').substring(0, 50))}... (${((f.first?.confidence || 0) * 100).toFixed(0)}%)</div>
                                <div class="fact-v2">C2: ${escapeHtml((f.second?.text || '').substring(0, 50))}... (${((f.second?.confidence || 0) * 100).toFixed(0)}%)</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }
}

function renderChainSummary(container, chain, label) {
    if (!container) return;

    container.innerHTML = `
        <h4>${label}: ${chain.id}</h4>
        <div class="chain-stats">
            <div class="chain-stat">
                <span class="stat-value">${chain.event_count}</span>
                <span class="stat-label">Events</span>
            </div>
            <div class="chain-stat">
                <span class="stat-value">${chain.agent_count}</span>
                <span class="stat-label">Agents</span>
            </div>
            <div class="chain-stat">
                <span class="stat-value">${chain.fact_count}</span>
                <span class="stat-label">Facts</span>
            </div>
            <div class="chain-stat ${chain.error_count > 0 ? 'has-errors' : ''}">
                <span class="stat-value">${chain.error_count}</span>
                <span class="stat-label">Errors</span>
            </div>
            <div class="chain-stat">
                <span class="stat-value">${formatDuration(chain.total_duration_ms)}</span>
                <span class="stat-label">Duration</span>
            </div>
            <div class="chain-stat">
                <span class="stat-value">${formatNumber(chain.total_tokens)}</span>
                <span class="stat-label">Tokens</span>
            </div>
        </div>
    `;
}

function formatDiffType(type) {
    const typeMap = {
        'diverged': 'Diverged',
        'missing_in_first': 'Missing in Chain 1',
        'missing_in_second': 'Missing in Chain 2'
    };
    return typeMap[type] || type;
}

// Metrics Loading and Rendering
async function loadMetrics() {
    if (!state.currentChain) return;

    try {
        const response = await fetch(`/api/metrics/${encodeURIComponent(state.currentChain.chain.filename)}`);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        evalState.metricsData = await response.json();
        renderMetricsDashboard();

    } catch (error) {
        console.error('Failed to load metrics:', error);
        showToast('Failed to load metrics', 'error');
    }
}

function renderMetricsDashboard() {
    const data = evalState.metricsData;
    if (!data) return;

    renderLatencyChart(data);
    renderTokenChart(data);
    renderAgentMetricsTable(data);
    renderErrorTimeline(data);
}

function renderLatencyChart(data) {
    const canvas = evalElements.latencyChart;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const agents = Object.keys(data.agent_metrics);
    const durations = agents.map(a => data.agent_metrics[a].duration_ms);

    if (evalState.latencyChartInstance) {
        evalState.latencyChartInstance = null;
    }

    const maxDuration = Math.max(...durations, 1);
    const barHeight = 30;
    const padding = 40;
    const labelWidth = 120;

    canvas.width = canvas.parentElement.clientWidth - 40;
    canvas.height = Math.max(agents.length * (barHeight + 10) + padding * 2, 200);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Agent Latency (ms)', canvas.width / 2, 20);

    agents.forEach((agent, i) => {
        const y = padding + i * (barHeight + 10);
        const duration = data.agent_metrics[agent].duration_ms;
        const barWidth = (duration / maxDuration) * (canvas.width - labelWidth - padding * 2);

        ctx.fillStyle = '#8b949e';
        ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(agent.substring(0, 15), labelWidth - 10, y + barHeight / 2 + 4);

        const gradient = ctx.createLinearGradient(labelWidth, y, labelWidth + barWidth, y);
        gradient.addColorStop(0, getAgentColor(agent));
        gradient.addColorStop(1, adjustColor(getAgentColor(agent), -20));

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(labelWidth, y, Math.max(barWidth, 2), barHeight, 4);
        ctx.fill();

        ctx.fillStyle = '#e6edf3';
        ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${formatDuration(duration)}`, labelWidth + barWidth + 8, y + barHeight / 2 + 4);
    });
}

function renderTokenChart(data) {
    const canvas = evalElements.tokenChart;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const tokensIn = data.token_distribution.input;
    const tokensOut = data.token_distribution.output;
    const total = tokensIn + tokensOut;

    canvas.width = canvas.parentElement.clientWidth - 40;
    canvas.height = 250;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2 + 10;
    const radius = Math.min(centerX, centerY) - 40;

    if (total > 0) {
        const inputAngle = (tokensIn / total) * 2 * Math.PI;

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, -Math.PI / 2, -Math.PI / 2 + inputAngle);
        ctx.closePath();
        ctx.fillStyle = '#58a6ff';
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, -Math.PI / 2 + inputAngle, -Math.PI / 2 + 2 * Math.PI);
        ctx.closePath();
        ctx.fillStyle = '#3fb950';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.6, 0, 2 * Math.PI);
        ctx.fillStyle = '#161b22';
        ctx.fill();

        ctx.fillStyle = '#e6edf3';
        ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(formatNumber(total), centerX, centerY - 5);
        ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = '#8b949e';
        ctx.fillText('Total Tokens', centerX, centerY + 12);
    } else {
        ctx.fillStyle = '#8b949e';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No token data', centerX, centerY);
    }

    const legendY = canvas.height - 25;
    ctx.fillStyle = '#58a6ff';
    ctx.fillRect(centerX - 100, legendY, 12, 12);
    ctx.fillStyle = '#e6edf3';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Input: ${formatNumber(tokensIn)}`, centerX - 82, legendY + 10);

    ctx.fillStyle = '#3fb950';
    ctx.fillRect(centerX + 20, legendY, 12, 12);
    ctx.fillStyle = '#e6edf3';
    ctx.fillText(`Output: ${formatNumber(tokensOut)}`, centerX + 38, legendY + 10);
}

function renderAgentMetricsTable(data) {
    const container = evalElements.agentMetricsTable;
    if (!container) return;

    const agents = Object.entries(data.agent_metrics);

    container.innerHTML = `
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Agent</th>
                    <th>Events</th>
                    <th>Duration</th>
                    <th>Tokens In</th>
                    <th>Tokens Out</th>
                    <th>Facts</th>
                    <th>Tools</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
                ${agents.map(([agent, metrics]) => `
                    <tr>
                        <td>
                            <span class="agent-indicator-small" style="background-color: ${getAgentColor(agent)}"></span>
                            ${agent}
                        </td>
                        <td>${metrics.event_count}</td>
                        <td>${formatDuration(metrics.duration_ms)}</td>
                        <td>${formatNumber(metrics.tokens_in)}</td>
                        <td>${formatNumber(metrics.tokens_out)}</td>
                        <td>${metrics.fact_count}</td>
                        <td>${metrics.tool_calls}</td>
                        <td class="${metrics.error_count > 0 ? 'has-errors' : ''}">${metrics.error_count}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderErrorTimeline(data) {
    const container = evalElements.errorTimeline;
    if (!container) return;

    if (data.error_timeline.length === 0) {
        container.innerHTML = '<div class="no-errors">No errors recorded</div>';
        return;
    }

    container.innerHTML = `
        <div class="error-timeline-list">
            ${data.error_timeline.map(error => `
                <div class="error-item">
                    <div class="error-marker"></div>
                    <div class="error-content">
                        <div class="error-header">
                            <span class="error-seq">#${error.seq}</span>
                            <span class="error-agent" style="color: ${getAgentColor(error.agent)}">${error.agent}</span>
                            <span class="error-category">${error.category}</span>
                            ${error.recoverable ? '<span class="error-recoverable">Recoverable</span>' : ''}
                        </div>
                        <div class="error-message">${escapeHtml(error.message)}</div>
                        <div class="error-time">${formatTimestamp(error.timestamp)}</div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// Evaluation Loading and Rendering
async function loadEvaluation() {
    if (!state.currentChain) return;

    try {
        const response = await fetch(`/api/evaluation/${encodeURIComponent(state.currentChain.chain.filename)}`);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        evalState.evaluationData = await response.json();
        renderEvaluationDashboard();

    } catch (error) {
        console.error('Failed to load evaluation:', error);
        showToast('Failed to load evaluation', 'error');
    }
}

function renderEvaluationDashboard() {
    const data = evalState.evaluationData;
    if (!data) return;

    renderScoreCards(data.scores);
    renderIssuesAndWarnings(data);
    renderConfidenceHeatmap(data);
    renderEvalSummary(data);
}

function renderScoreCards(scores) {
    if (evalElements.overallScore) {
        evalElements.overallScore.textContent = scores.overall;
        evalElements.overallScore.className = `score-value ${getScoreClass(scores.overall)}`;
    }

    if (evalElements.overallRing) {
        evalElements.overallRing.setAttribute('stroke-dasharray', `${scores.overall}, 100`);
        evalElements.overallRing.style.stroke = getScoreColor(scores.overall);
    }

    if (evalElements.perfScore) {
        evalElements.perfScore.textContent = scores.performance;
    }
    if (evalElements.perfBar) {
        evalElements.perfBar.style.width = `${scores.performance}%`;
        evalElements.perfBar.style.backgroundColor = getScoreColor(scores.performance);
    }

    if (evalElements.reliabilityScore) {
        evalElements.reliabilityScore.textContent = scores.reliability;
    }
    if (evalElements.reliabilityBar) {
        evalElements.reliabilityBar.style.width = `${scores.reliability}%`;
        evalElements.reliabilityBar.style.backgroundColor = getScoreColor(scores.reliability);
    }

    if (evalElements.qualityScore) {
        evalElements.qualityScore.textContent = scores.quality;
    }
    if (evalElements.qualityBar) {
        evalElements.qualityBar.style.width = `${scores.quality}%`;
        evalElements.qualityBar.style.backgroundColor = getScoreColor(scores.quality);
    }
}

function renderIssuesAndWarnings(data) {
    const container = evalElements.issuesList;
    if (!container) return;

    const allItems = [...data.issues, ...data.warnings];

    if (allItems.length === 0) {
        container.innerHTML = '<div class="no-issues">No issues or warnings detected</div>';
        return;
    }

    container.innerHTML = allItems.map(item => `
        <div class="issue-item severity-${item.severity}">
            <div class="issue-icon">
                ${item.severity === 'high' ? '!' : item.severity === 'medium' ? '!!' : 'i'}
            </div>
            <div class="issue-content">
                <div class="issue-type">${item.type.replace(/_/g, ' ')}</div>
                <div class="issue-message">${escapeHtml(item.message)}</div>
            </div>
            <div class="issue-severity">${item.severity}</div>
        </div>
    `).join('');
}

function renderConfidenceHeatmap(data) {
    const container = evalElements.confidenceHeatmap;
    if (!container) return;

    const heatmapData = data.fact_confidence_heatmap;

    if (heatmapData.length === 0) {
        container.innerHTML = '<div class="no-heatmap">No fact confidence data available</div>';
        return;
    }

    const factIds = [...new Set(heatmapData.map(d => d.fact_id))];
    const maxSeq = Math.max(...heatmapData.map(d => d.seq));

    container.innerHTML = `
        <div class="heatmap-grid">
            <div class="heatmap-header">
                <div class="heatmap-corner">Fact ID</div>
                <div class="heatmap-seqs">
                    ${Array.from({length: Math.min(maxSeq, 20)}, (_, i) => i + 1).map(seq => `
                        <div class="heatmap-seq">${seq}</div>
                    `).join('')}
                </div>
            </div>
            <div class="heatmap-body">
                ${factIds.slice(0, 10).map(factId => {
                    const factData = heatmapData.filter(d => d.fact_id === factId);
                    return `
                        <div class="heatmap-row">
                            <div class="heatmap-label">${factId.substring(0, 12)}</div>
                            <div class="heatmap-cells">
                                ${Array.from({length: Math.min(maxSeq, 20)}, (_, i) => {
                                    const point = factData.find(d => d.seq === i + 1);
                                    if (point) {
                                        return `<div class="heatmap-cell" style="background-color: ${getConfidenceColor(point.confidence)}" title="${factId}: ${(point.confidence * 100).toFixed(0)}% at seq ${point.seq}"></div>`;
                                    }
                                    return '<div class="heatmap-cell empty"></div>';
                                }).join('')}
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
        <div class="heatmap-legend">
            <span>Low</span>
            <div class="heatmap-legend-bar"></div>
            <span>High</span>
        </div>
    `;
}

function renderEvalSummary(data) {
    const container = evalElements.evalSummary;
    if (!container) return;

    const metrics = data.metrics;

    container.innerHTML = `
        <div class="eval-metrics-grid">
            <div class="eval-metric">
                <div class="eval-metric-value">${metrics.total_events}</div>
                <div class="eval-metric-label">Total Events</div>
            </div>
            <div class="eval-metric">
                <div class="eval-metric-value">${formatDuration(metrics.total_duration_ms)}</div>
                <div class="eval-metric-label">Total Duration</div>
            </div>
            <div class="eval-metric">
                <div class="eval-metric-value">${formatNumber(metrics.total_tokens)}</div>
                <div class="eval-metric-label">Total Tokens</div>
            </div>
            <div class="eval-metric">
                <div class="eval-metric-value">${metrics.fact_count}</div>
                <div class="eval-metric-label">Facts Generated</div>
            </div>
            <div class="eval-metric">
                <div class="eval-metric-value">${(metrics.average_confidence * 100).toFixed(1)}%</div>
                <div class="eval-metric-label">Avg Confidence</div>
            </div>
            <div class="eval-metric ${metrics.error_count > 0 ? 'has-errors' : ''}">
                <div class="eval-metric-value">${metrics.error_count}</div>
                <div class="eval-metric-label">Errors</div>
            </div>
        </div>
        ${data.low_confidence_facts.length > 0 ? `
            <div class="low-confidence-section">
                <h4>Low Confidence Facts (< 70%)</h4>
                <div class="low-confidence-list">
                    ${data.low_confidence_facts.slice(0, 5).map(fact => `
                        <div class="low-confidence-fact">
                            <span class="fact-id">${fact.id}</span>
                            <span class="fact-confidence" style="color: ${getConfidenceColor(fact.confidence)}">${(fact.confidence * 100).toFixed(0)}%</span>
                            <span class="fact-text">${escapeHtml((fact.text || '').substring(0, 50))}...</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        ` : ''}
    `;
}

// Utility Functions for Evaluation
function getScoreClass(score) {
    if (score >= 80) return 'score-good';
    if (score >= 60) return 'score-medium';
    return 'score-poor';
}

function getScoreColor(score) {
    if (score >= 80) return '#3fb950';
    if (score >= 60) return '#d29922';
    return '#f85149';
}

function getConfidenceColor(confidence) {
    const r = Math.round(248 - confidence * 185);
    const g = Math.round(81 + confidence * 104);
    const b = Math.round(73 + confidence * 7);
    return `rgb(${r}, ${g}, ${b})`;
}

function adjustColor(color, amount) {
    const hex = color.replace('#', '');
    const r = Math.max(0, Math.min(255, parseInt(hex.substr(0, 2), 16) + amount));
    const g = Math.max(0, Math.min(255, parseInt(hex.substr(2, 2), 16) + amount));
    const b = Math.max(0, Math.min(255, parseInt(hex.substr(4, 2), 16) + amount));
    return `rgb(${r}, ${g}, ${b})`;
}

// Override loadChain to enable compare button
const originalLoadChain = loadChain;
loadChain = async function(filename) {
    await originalLoadChain(filename);
    if (evalElements.compareBtn) {
        evalElements.compareBtn.disabled = state.chains.length < 2;
    }
};

// Override loadChains to enable compare button
const originalLoadChains = loadChains;
loadChains = async function() {
    await originalLoadChains();
    if (evalElements.compareBtn) {
        evalElements.compareBtn.disabled = state.chains.filter(c => !c.error).length < 2;
    }
};

// ============================================================================
// WEBSOCKET STREAMING FEATURES
// ============================================================================

function connectWebSocket() {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
        state.websocket = new WebSocket(wsUrl);

        state.websocket.onopen = function() {
            state.wsConnected = true;
            console.log('[LCTL] WebSocket connected');
            updateConnectionStatus(true);
        };

        state.websocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleStreamingEvent(data);
            } catch (e) {
                console.error('[LCTL] Failed to parse WebSocket message:', e);
            }
        };

        state.websocket.onclose = function() {
            state.wsConnected = false;
            state.wsClientId = null;
            console.log('[LCTL] WebSocket disconnected');
            updateConnectionStatus(false);

            if (state.liveMode) {
                setTimeout(connectWebSocket, 3000);
            }
        };

        state.websocket.onerror = function(error) {
            console.error('[LCTL] WebSocket error:', error);
            updateConnectionStatus(false);
        };

    } catch (error) {
        console.error('[LCTL] Failed to create WebSocket:', error);
    }
}

function disconnectWebSocket() {
    if (state.websocket) {
        state.websocket.close();
        state.websocket = null;
    }
    state.wsConnected = false;
    state.wsClientId = null;
}

function handleStreamingEvent(data) {
    switch (data.type) {
        case 'connected':
            state.wsClientId = data.client_id;
            showToast(`Connected to stream (ID: ${data.client_id})`, 'success');
            break;

        case 'heartbeat':
            break;

        case 'pong':
            break;

        case 'event':
            handleLiveEvent(data);
            break;

        case 'chain_start':
            handleChainStart(data);
            break;

        case 'chain_end':
            handleChainEnd(data);
            break;

        case 'error':
            console.error('[LCTL] Stream error:', data.payload?.message);
            showToast(`Stream error: ${data.payload?.message || 'Unknown'}`, 'error');
            break;

        default:
            console.log('[LCTL] Unknown streaming event:', data.type);
    }
}

function handleLiveEvent(data) {
    if (!state.liveMode) return;

    const payload = data.payload;
    if (!payload) return;

    state.liveEvents.push(payload);

    if (state.liveEvents.length > 1000) {
        state.liveEvents = state.liveEvents.slice(-500);
    }

    if (state.currentChain && data.chain_id === state.currentChain.chain.id) {
        state.currentChain.events.push(payload);
        state.maxSeq = payload.seq;
        state.currentSeq = state.maxSeq;

        updateSlider();
        renderTimeline();
        renderSwimlanes();

        if (payload.type === 'fact_added' || payload.type === 'fact_modified') {
            renderFactRegistry();
        }

        updateLiveStats(payload);
    }

    addLiveEventToTimeline(payload);
}

function handleChainStart(data) {
    if (state.liveMode) {
        showToast(`New chain started: ${data.chain_id || data.payload?.chain_id}`, 'info');
    }
}

function handleChainEnd(data) {
    if (state.liveMode) {
        const chainId = data.chain_id || data.payload?.chain_id;
        const eventCount = data.payload?.event_count || 0;
        showToast(`Chain ended: ${chainId} (${eventCount} events)`, 'info');

        loadChains();
    }
}

function addLiveEventToTimeline(event) {
    const container = elements.timeline;
    if (!container) return;

    const el = document.createElement('div');
    el.className = 'timeline-event live-event';
    el.setAttribute('data-seq', event.seq);

    el.innerHTML = `
        <span class="event-seq">#${event.seq}</span>
        <div class="event-info">
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="event-type-badge ${event.type}">${formatEventType(event.type)}</span>
                <span class="event-agent" style="color: ${getAgentColor(event.agent)}">${event.agent}</span>
                <span class="live-indicator">LIVE</span>
            </div>
            <div class="event-summary">${getEventSummary(event)}</div>
        </div>
    `;

    el.addEventListener('click', () => selectEvent(event));
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
}

function updateLiveStats(event) {
    if (!state.currentChain) return;

    const eventType = event.type;

    if (eventType === 'step_end') {
        const duration = event.data?.duration_ms || 0;
        const tokensIn = event.data?.tokens?.input || event.data?.tokens?.in || 0;
        const tokensOut = event.data?.tokens?.output || event.data?.tokens?.out || 0;

        state.currentChain.state.metrics.total_duration_ms += duration;
        state.currentChain.state.metrics.total_tokens_in += tokensIn;
        state.currentChain.state.metrics.total_tokens_out += tokensOut;
    }

    if (eventType === 'error') {
        state.currentChain.state.metrics.error_count += 1;
    }

    if (eventType === 'fact_added') {
        const factId = event.data?.id;
        if (factId) {
            state.currentChain.state.facts[factId] = {
                text: event.data?.text || '',
                confidence: event.data?.confidence || 1.0,
                source: event.data?.source || event.agent
            };
        }
    }

    if (eventType === 'fact_modified') {
        const factId = event.data?.id;
        if (factId && state.currentChain.state.facts[factId]) {
            if (event.data?.text) {
                state.currentChain.state.facts[factId].text = event.data.text;
            }
            if (event.data?.confidence !== undefined) {
                state.currentChain.state.facts[factId].confidence = event.data.confidence;
            }
        }
    }

    updateStats();
}

function toggleLiveMode() {
    state.liveMode = !state.liveMode;

    if (state.liveMode) {
        connectWebSocket();
        showToast('Live mode enabled', 'success');
        updateLiveModeUI(true);
    } else {
        disconnectWebSocket();
        showToast('Live mode disabled', 'info');
        updateLiveModeUI(false);
    }
}

function updateLiveModeUI(isLive) {
    const liveBtn = document.getElementById('live-btn');
    if (liveBtn) {
        liveBtn.classList.toggle('active', isLive);
        liveBtn.textContent = isLive ? 'LIVE' : 'Go Live';
    }

    const liveIndicator = document.getElementById('live-indicator');
    if (liveIndicator) {
        liveIndicator.classList.toggle('hidden', !isLive);
    }
}

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    if (statusEl) {
        statusEl.classList.toggle('connected', connected);
        statusEl.classList.toggle('disconnected', !connected);
        statusEl.title = connected ? 'Connected to live stream' : 'Disconnected';
    }
}

function subscribeToChain(chainId) {
    if (!state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
        console.warn('[LCTL] WebSocket not connected');
        return;
    }

    state.websocket.send(JSON.stringify({
        type: 'subscribe',
        filters: {
            chain_id: chainId
        }
    }));
}

function subscribeToEventTypes(eventTypes) {
    if (!state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
        console.warn('[LCTL] WebSocket not connected');
        return;
    }

    state.websocket.send(JSON.stringify({
        type: 'subscribe',
        filters: {
            event_types: eventTypes
        }
    }));
}

function unsubscribeFromStream() {
    if (!state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
        return;
    }

    state.websocket.send(JSON.stringify({
        type: 'unsubscribe'
    }));
}

function sendPing() {
    if (!state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
        return;
    }

    state.websocket.send(JSON.stringify({
        type: 'ping'
    }));
}

async function getStreamingStatus() {
    try {
        const response = await fetch('/api/streaming/status');
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('[LCTL] Failed to get streaming status:', error);
        return null;
    }
}

window.lctlStreaming = {
    connect: connectWebSocket,
    disconnect: disconnectWebSocket,
    toggleLive: toggleLiveMode,
    subscribe: subscribeToChain,
    subscribeEventTypes: subscribeToEventTypes,
    unsubscribe: unsubscribeFromStream,
    ping: sendPing,
    getStatus: getStreamingStatus,
    get isConnected() { return state.wsConnected; },
    get isLive() { return state.liveMode; },
    get clientId() { return state.wsClientId; }
};
