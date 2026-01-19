import * as vscode from 'vscode';
import * as path from 'path';
import { ChainTreeProvider, ChainTreeItem, LctlChainFile, LctlEvent } from './chainProvider';

type DashboardOpener = (uri?: vscode.Uri) => void;

export function registerCommands(
    context: vscode.ExtensionContext,
    chainProvider: ChainTreeProvider,
    openDashboard: DashboardOpener
): void {
    // Open Dashboard
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.openDashboard', () => {
            openDashboard();
        })
    );

    // Load Chain
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.loadChain', async (arg?: vscode.Uri | ChainTreeItem) => {
            let uri: vscode.Uri | undefined;

            if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            } else {
                const files = await vscode.workspace.findFiles('**/*.lctl.json', '**/node_modules/**');
                if (files.length === 0) {
                    void vscode.window.showWarningMessage('No LCTL chain files found in workspace.');
                    return;
                }

                const items = files.map((file) => ({
                    label: vscode.workspace.asRelativePath(file),
                    uri: file
                }));

                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select an LCTL chain file to load'
                });

                if (selected) {
                    uri = selected.uri;
                }
            }

            if (uri) {
                openDashboard(uri);
            }
        })
    );

    // Replay Chain
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.replay', async (arg?: string | vscode.Uri | ChainTreeItem) => {
            let uri: vscode.Uri | undefined;

            if (typeof arg === 'string') {
                uri = chainProvider.findChainUri(arg);
            } else if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            }

            if (!uri) {
                void vscode.window.showErrorMessage('Could not find chain file for replay.');
                return;
            }

            const terminalName = 'LCTL Replay';
            let terminal = vscode.window.terminals.find(t => t.name === terminalName);
            if (!terminal) {
                terminal = vscode.window.createTerminal(terminalName);
            }

            terminal.show();
            terminal.sendText(`lctl replay "${uri.fsPath}"`);
        })
    );

    // Export HTML Report
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.exportHtml', async (arg?: vscode.Uri | ChainTreeItem) => {
            let uri: vscode.Uri | undefined;

            if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            }

            if (!uri) {
                void vscode.window.showErrorMessage('No chain selected for export.');
                return;
            }

            const chainData = await chainProvider.getChainData(uri);
            if (!chainData) {
                void vscode.window.showErrorMessage('Could not load chain data.');
                return;
            }

            // Generate HTML report
            const html = generateHtmlReport(chainData, uri.fsPath);

            // Ask user where to save
            const defaultUri = vscode.Uri.file(
                uri.fsPath.replace('.lctl.json', '-report.html')
            );

            const saveUri = await vscode.window.showSaveDialog({
                defaultUri,
                filters: { 'HTML Files': ['html'] },
                title: 'Save HTML Report'
            });

            if (saveUri) {
                await vscode.workspace.fs.writeFile(saveUri, Buffer.from(html, 'utf8'));

                const open = await vscode.window.showInformationMessage(
                    'HTML report exported successfully.',
                    'Open in Browser'
                );

                if (open) {
                    await vscode.env.openExternal(saveUri);
                }
            }
        })
    );

    // Export Markdown Report
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.exportMarkdown', async (arg?: vscode.Uri | ChainTreeItem) => {
            let uri: vscode.Uri | undefined;

            if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            }

            if (!uri) {
                void vscode.window.showErrorMessage('No chain selected for export.');
                return;
            }

            const chainData = await chainProvider.getChainData(uri);
            if (!chainData) {
                void vscode.window.showErrorMessage('Could not load chain data.');
                return;
            }

            // Generate Markdown report
            const markdown = generateMarkdownReport(chainData, uri.fsPath);

            // Ask user where to save
            const defaultUri = vscode.Uri.file(
                uri.fsPath.replace('.lctl.json', '-report.md')
            );

            const saveUri = await vscode.window.showSaveDialog({
                defaultUri,
                filters: { 'Markdown Files': ['md'] },
                title: 'Save Markdown Report'
            });

            if (saveUri) {
                await vscode.workspace.fs.writeFile(saveUri, Buffer.from(markdown, 'utf8'));

                const open = await vscode.window.showInformationMessage(
                    'Markdown report exported successfully.',
                    'Open File'
                );

                if (open) {
                    await vscode.window.showTextDocument(saveUri);
                }
            }
        })
    );

    // Show Statistics
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.showStats', async (arg?: vscode.Uri | ChainTreeItem) => {
            let uri: vscode.Uri | undefined;

            if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            }

            if (!uri) {
                void vscode.window.showErrorMessage('No chain selected.');
                return;
            }

            const terminalName = 'LCTL Stats';
            let terminal = vscode.window.terminals.find(t => t.name === terminalName);
            if (!terminal) {
                terminal = vscode.window.createTerminal(terminalName);
            }

            terminal.show();
            terminal.sendText(`lctl stats "${uri.fsPath}"`);
        })
    );

    // Compare Chains
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.compareChains', async (arg?: vscode.Uri | ChainTreeItem) => {
            let firstUri: vscode.Uri | undefined;

            if (arg instanceof vscode.Uri) {
                firstUri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                firstUri = arg.resourceUri;
            }

            // Get all chain files
            const files = await vscode.workspace.findFiles('**/*.lctl.json', '**/node_modules/**');

            if (files.length < 2) {
                void vscode.window.showWarningMessage('Need at least 2 chain files to compare.');
                return;
            }

            // If no first file selected, let user pick
            if (!firstUri) {
                const items = files.map((file) => ({
                    label: path.basename(file.fsPath, '.lctl.json'),
                    description: vscode.workspace.asRelativePath(file),
                    uri: file
                }));

                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select first chain to compare'
                });

                if (!selected) {return;}
                firstUri = selected.uri;
            }

            // Now pick second file
            const otherFiles = files.filter(f => f.fsPath !== firstUri!.fsPath);
            const items = otherFiles.map((file) => ({
                label: path.basename(file.fsPath, '.lctl.json'),
                description: vscode.workspace.asRelativePath(file),
                uri: file
            }));

            const second = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select second chain to compare'
            });

            if (!second) {return;}

            // Run diff command
            const terminalName = 'LCTL Diff';
            let terminal = vscode.window.terminals.find(t => t.name === terminalName);
            if (!terminal) {
                terminal = vscode.window.createTerminal(terminalName);
            }

            terminal.show();
            terminal.sendText(`lctl diff "${firstUri.fsPath}" "${second.uri.fsPath}"`);
        })
    );
}

/**
 * Escape HTML special characters to prevent XSS
 */
function escapeHtml(text: string): string {
    const htmlEntities: Record<string, string> = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    };
    return text.replace(/[&<>"']/g, char => htmlEntities[char] || char);
}

/**
 * Generate a standalone HTML report for a chain
 */
function generateHtmlReport(chainData: LctlChainFile, filePath: string): string {
    const chainId = escapeHtml(chainData.chain?.id || 'Unknown Chain');
    const events = chainData.events || [];

    // Compute stats
    let totalDurationMs = 0;
    let tokensIn = 0;
    let tokensOut = 0;
    let errorCount = 0;
    const agents = new Set<string>();

    for (const event of events) {
        if (event.agent) {agents.add(event.agent);}
        if (event.type === 'error') {errorCount++;}
        if (event.type === 'step_end' && event.data) {
            totalDurationMs += Number(event.data.duration_ms) || 0;
            tokensIn += Number(event.data.tokens_in) || 0;
            tokensOut += Number(event.data.tokens_out) || 0;
        }
    }

    // Valid event types for CSS class sanitization
    const validEventTypes = ['step_start', 'step_end', 'fact_added', 'fact_modified', 'tool_call', 'error', 'checkpoint'];

    const eventRows = events.map(event => {
        const time = new Date(event.timestamp).toLocaleTimeString();
        const icon = getEventIcon(event.type);
        const details = escapeHtml(formatEventDetails(event));
        const safeType = validEventTypes.includes(event.type) ? event.type : 'unknown';
        const safeAgent = escapeHtml(event.agent || '-');
        return `
            <tr class="event-${safeType}">
                <td>${event.seq}</td>
                <td>${icon} ${escapeHtml(event.type)}</td>
                <td>${safeAgent}</td>
                <td>${time}</td>
                <td><pre>${details}</pre></td>
            </tr>
        `;
    }).join('\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCTL Report: ${chainId}</title>
    <style>
        :root {
            --bg: #1e1e1e;
            --fg: #d4d4d4;
            --accent: #007acc;
            --success: #4ec9b0;
            --error: #f14c4c;
            --warning: #cca700;
            --border: #3c3c3c;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--fg);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 10px; }
        h2 { color: var(--fg); margin-top: 30px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #252526;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value { font-size: 24px; font-weight: bold; color: var(--accent); }
        .stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: #252526; font-weight: 600; }
        tr:hover { background: #2a2d2e; }
        pre { margin: 0; font-size: 12px; white-space: pre-wrap; max-width: 400px; overflow: hidden; }
        .event-step_start { border-left: 3px solid var(--accent); }
        .event-step_end { border-left: 3px solid var(--success); }
        .event-error { border-left: 3px solid var(--error); background: rgba(241, 76, 76, 0.1); }
        .event-tool_call { border-left: 3px solid var(--warning); }
        .event-fact_added { border-left: 3px solid #9b59b6; }
        .meta { color: #888; font-size: 12px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 LCTL Report: ${chainId}</h1>
        <p class="meta">
            Generated: ${new Date().toLocaleString()} |
            Source: ${path.basename(filePath)} |
            LCTL Version: ${chainData.lctl}
        </p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">${events.length}</div>
                <div class="stat-label">Events</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${agents.size}</div>
                <div class="stat-label">Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(totalDurationMs / 1000).toFixed(1)}s</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${tokensIn.toLocaleString()}</div>
                <div class="stat-label">Tokens In</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${tokensOut.toLocaleString()}</div>
                <div class="stat-label">Tokens Out</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${errorCount}</div>
                <div class="stat-label">Errors</div>
            </div>
        </div>

        <h2>📋 Event Log</h2>
        <table>
            <thead>
                <tr>
                    <th>Seq</th>
                    <th>Type</th>
                    <th>Agent</th>
                    <th>Time</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                ${eventRows}
            </tbody>
        </table>
    </div>
</body>
</html>`;
}

function getEventIcon(type: string): string {
    const icons: Record<string, string> = {
        'step_start': '▶️',
        'step_end': '⏹️',
        'tool_call': '🔧',
        'fact_added': '💡',
        'fact_modified': '✏️',
        'error': '❌',
        'checkpoint': '📍',
    };
    return icons[type] || '•';
}

function formatEventDetails(event: LctlEvent): string {
    if (!event.data) {return '-';}

    const data = event.data;
    const lines: string[] = [];

    if (event.type === 'step_start') {
        if (data.intent) {lines.push(`Intent: ${data.intent}`);}
        if (data.input_summary) {lines.push(`Input: ${truncate(String(data.input_summary), 100)}`);}
    } else if (event.type === 'step_end') {
        if (data.outcome) {lines.push(`Outcome: ${data.outcome}`);}
        if (data.duration_ms) {lines.push(`Duration: ${data.duration_ms}ms`);}
        if (data.tokens_in || data.tokens_out) {
            lines.push(`Tokens: ${data.tokens_in || 0} in / ${data.tokens_out || 0} out`);
        }
    } else if (event.type === 'tool_call') {
        if (data.tool) {lines.push(`Tool: ${data.tool}`);}
        if (data.duration_ms) {lines.push(`Duration: ${data.duration_ms}ms`);}
    } else if (event.type === 'fact_added' || event.type === 'fact_modified') {
        if (data.id) {lines.push(`Fact: ${data.id}`);}
        if (data.text) {lines.push(`Text: ${truncate(String(data.text), 100)}`);}
        if (data.confidence !== undefined && data.confidence !== null) {lines.push(`Confidence: ${(Number(data.confidence) * 100).toFixed(0)}%`);}
    } else if (event.type === 'error') {
        if (data.error_type) {lines.push(`Type: ${data.error_type}`);}
        if (data.message) {lines.push(`Message: ${truncate(String(data.message), 100)}`);}
    }

    return lines.join('\n') || JSON.stringify(data, null, 2).slice(0, 200);
}

function truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) {return text;}
    return text.slice(0, maxLen - 3) + '...';
}

/**
 * Generate a Markdown report for a chain
 */
function generateMarkdownReport(chainData: LctlChainFile, filePath: string): string {
    const chainId = chainData.chain?.id || 'Unknown Chain';
    const events = chainData.events || [];

    // Compute stats
    let totalDurationMs = 0;
    let tokensIn = 0;
    let tokensOut = 0;
    let errorCount = 0;
    const agents = new Set<string>();
    const tools = new Set<string>();

    for (const event of events) {
        if (event.agent) {agents.add(event.agent);}
        if (event.type === 'error') {errorCount++;}
        if (event.type === 'tool_call' && event.data?.tool) {
            tools.add(String(event.data.tool));
        }
        if (event.type === 'step_end' && event.data) {
            totalDurationMs += Number(event.data.duration_ms) || 0;
            tokensIn += Number(event.data.tokens_in) || 0;
            tokensOut += Number(event.data.tokens_out) || 0;
        }
    }

    const lines: string[] = [
        `# LCTL Report: ${chainId}`,
        '',
        `> Generated: ${new Date().toLocaleString()}`,
        `> Source: \`${path.basename(filePath)}\``,
        `> LCTL Version: ${chainData.lctl}`,
        '',
        '## Summary',
        '',
        '| Metric | Value |',
        '|--------|-------|',
        `| Events | ${events.length} |`,
        `| Agents | ${agents.size} |`,
        `| Duration | ${(totalDurationMs / 1000).toFixed(1)}s |`,
        `| Tokens In | ${tokensIn.toLocaleString()} |`,
        `| Tokens Out | ${tokensOut.toLocaleString()} |`,
        `| Errors | ${errorCount} |`,
        `| Tools Used | ${tools.size} |`,
        '',
    ];

    // Agents section
    if (agents.size > 0) {
        lines.push('## Agents', '', '| Agent | Events |', '|-------|--------|');
        const agentCounts = new Map<string, number>();
        for (const event of events) {
            if (event.agent) {
                agentCounts.set(event.agent, (agentCounts.get(event.agent) || 0) + 1);
            }
        }
        for (const [agent, count] of agentCounts) {
            lines.push(`| ${agent} | ${count} |`);
        }
        lines.push('');
    }

    // Tools section
    if (tools.size > 0) {
        lines.push('## Tools Used', '', '| Tool | Count |', '|------|-------|');
        const toolCounts = new Map<string, number>();
        for (const event of events) {
            if (event.type === 'tool_call' && event.data?.tool) {
                const tool = event.data.tool as string;
                toolCounts.set(tool, (toolCounts.get(tool) || 0) + 1);
            }
        }
        for (const [tool, count] of toolCounts) {
            lines.push(`| ${tool} | ${count} |`);
        }
        lines.push('');
    }

    // Event log section
    lines.push('## Event Log', '');

    for (const event of events) {
        const icon = getEventIcon(event.type);
        const time = new Date(event.timestamp).toLocaleTimeString();
        const agentTag = event.agent ? ` [${event.agent}]` : '';

        lines.push(`### ${icon} [${event.seq}] ${event.type}${agentTag}`);
        lines.push(`*${time}*`);
        lines.push('');

        if (event.data) {
            const details = formatMarkdownEventDetails(event);
            if (details) {
                lines.push(details);
                lines.push('');
            }
        }
    }

    return lines.join('\n');
}

function formatMarkdownEventDetails(event: LctlEvent): string {
    const data = event.data;
    if (!data) {return '';}

    const lines: string[] = [];

    if (event.type === 'step_start') {
        if (data.intent) {lines.push(`- **Intent**: ${data.intent}`);}
        if (data.input_summary) {lines.push(`- **Input**: ${truncate(data.input_summary as string, 150)}`);}
    } else if (event.type === 'step_end') {
        if (data.outcome) {lines.push(`- **Outcome**: ${data.outcome}`);}
        if (data.duration_ms) {lines.push(`- **Duration**: ${data.duration_ms}ms`);}
        if (data.tokens_in || data.tokens_out) {
            lines.push(`- **Tokens**: ${data.tokens_in || 0} in / ${data.tokens_out || 0} out`);
        }
        if (data.output_summary) {lines.push(`- **Output**: ${truncate(data.output_summary as string, 150)}`);}
    } else if (event.type === 'tool_call') {
        if (data.tool) {lines.push(`- **Tool**: \`${data.tool}\``);}
        if (data.duration_ms) {lines.push(`- **Duration**: ${data.duration_ms}ms`);}
    } else if (event.type === 'fact_added' || event.type === 'fact_modified') {
        if (data.id) {lines.push(`- **Fact ID**: ${data.id}`);}
        if (data.confidence !== undefined) {lines.push(`- **Confidence**: ${((data.confidence as number) * 100).toFixed(0)}%`);}
        if (data.text) {lines.push(`- **Text**: ${truncate(data.text as string, 150)}`);}
    } else if (event.type === 'error') {
        if (data.error_type) {lines.push(`- **Error Type**: ${data.error_type}`);}
        if (data.message) {lines.push(`- **Message**: ${truncate(data.message as string, 150)}`);}
    }

    return lines.join('\n');
}
