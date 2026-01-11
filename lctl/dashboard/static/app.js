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
    eventFilter: 'all'
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
