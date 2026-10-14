import * as vscode from 'vscode';
import * as path from 'path';
import { chainCost, chainTokens, limitEvents, pricingFor } from './chainStats';

// LCTL v4.0 Schema Types
export interface LctlChainFile {
    lctl: string;  // Version like "4.0"
    chain: {
        id: string;
    };
    events: LctlEvent[];
}

export interface LctlEvent {
    seq: number;
    type: string;
    timestamp: string;
    agent?: string;
    data?: Record<string, unknown>;
}

// Computed statistics for a chain
export interface ChainStats {
    eventCount: number;
    errorCount: number;
    warningCount: number;
    totalDurationMs: number;
    tokensIn: number;
    tokensOut: number;
    estimatedCost: number;
    isRecording: boolean;
    agents: Set<string>;
    lastModified: Date;
}

/** Model chosen in `lctl.defaultPricing`, used for cost estimates. */
export function configuredModel(): string | undefined {
    return vscode.workspace.getConfiguration('lctl').get<string>('defaultPricing');
}

export class ChainTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly resourceUri?: vscode.Uri,
        public readonly chainData?: LctlChainFile,
        public readonly stats?: ChainStats,
        public readonly event?: LctlEvent
    ) {
        super(label, collapsibleState);

        if (resourceUri && !event) {
            // Chain file item
            this.contextValue = 'chainFile';
            this.iconPath = this.getChainIcon();
            this.tooltip = this.getChainTooltip();
            this.description = this.getChainDescription();
            this.command = {
                command: 'lctl.loadChain',
                title: 'Load Chain',
                arguments: [resourceUri]
            };
        } else if (event) {
            // Event item
            this.contextValue = 'event';
            this.iconPath = this.getEventIcon();
            this.tooltip = this.getEventTooltip();
            this.description = event.agent || '';
        }
    }

    private getChainIcon(): vscode.ThemeIcon {
        if (!this.stats) {
            return new vscode.ThemeIcon('git-branch');
        }

        if (this.stats.isRecording) {
            return new vscode.ThemeIcon('record', new vscode.ThemeColor('charts.red'));
        }
        if (this.stats.errorCount > 0) {
            return new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
        }
        return new vscode.ThemeIcon('check', new vscode.ThemeColor('charts.green'));
    }

    private getChainDescription(): string {
        if (!this.stats) {
            return '';
        }

        const parts: string[] = [];

        // Event count
        parts.push(`${this.stats.eventCount} events`);

        // Cost if available
        if (this.stats.estimatedCost > 0) {
            parts.push(`$${this.stats.estimatedCost.toFixed(3)}`);
        }

        // Error indicator
        if (this.stats.errorCount > 0) {
            parts.push(`⚠️${this.stats.errorCount}`);
        }

        return parts.join('  ');
    }

    private getChainTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.isTrusted = true;

        const chainId = this.chainData?.chain?.id || this.label;
        md.appendMarkdown(`**${chainId}**\n\n`);

        if (this.stats) {
            md.appendMarkdown(`| Metric | Value |\n`);
            md.appendMarkdown(`|--------|-------|\n`);
            md.appendMarkdown(`| Events | ${this.stats.eventCount} |\n`);
            md.appendMarkdown(`| Agents | ${this.stats.agents.size} |\n`);

            if (this.stats.totalDurationMs > 0) {
                const duration = (this.stats.totalDurationMs / 1000).toFixed(1);
                md.appendMarkdown(`| Duration | ${duration}s |\n`);
            }

            if (this.stats.tokensIn > 0 || this.stats.tokensOut > 0) {
                md.appendMarkdown(`| Tokens In | ${this.stats.tokensIn.toLocaleString()} |\n`);
                md.appendMarkdown(`| Tokens Out | ${this.stats.tokensOut.toLocaleString()} |\n`);
            }

            if (this.stats.estimatedCost > 0) {
                md.appendMarkdown(`| Est. Cost | $${this.stats.estimatedCost.toFixed(4)} |\n`);
            }

            if (this.stats.errorCount > 0) {
                md.appendMarkdown(`| Errors | ${this.stats.errorCount} |\n`);
            }

            md.appendMarkdown(`\n*Last updated: ${this.stats.lastModified.toLocaleString()}*`);
        }

        return md;
    }

    private getEventIcon(): vscode.ThemeIcon {
        if (!this.event) {
            return new vscode.ThemeIcon('circle-outline');
        }

        const iconMap: Record<string, [string, string?]> = {
            'step_start': ['debug-start', 'charts.blue'],
            'step_end': ['debug-stop', 'charts.green'],
            'tool_call': ['tools', 'charts.yellow'],
            'fact_added': ['lightbulb', 'charts.purple'],
            'fact_modified': ['edit', 'charts.purple'],
            'error': ['error', 'errorForeground'],
            'checkpoint': ['bookmark', 'charts.orange'],
        };

        const [icon, color] = iconMap[this.event.type] || ['circle-outline', undefined];
        return color
            ? new vscode.ThemeIcon(icon, new vscode.ThemeColor(color))
            : new vscode.ThemeIcon(icon);
    }

    private getEventTooltip(): string {
        if (!this.event) {
            return '';
        }

        const lines = [
            `Type: ${this.event.type}`,
            `Seq: ${this.event.seq}`,
            `Time: ${new Date(this.event.timestamp).toLocaleString()}`,
        ];

        if (this.event.agent) {
            lines.push(`Agent: ${this.event.agent}`);
        }

        if (this.event.data) {
            if (this.event.data.tool) {
                lines.push(`Tool: ${this.event.data.tool}`);
            }
            if (this.event.data.duration_ms) {
                lines.push(`Duration: ${this.event.data.duration_ms}ms`);
            }
        }

        return lines.join('\n');
    }
}

export class ChainTreeProvider implements vscode.TreeDataProvider<ChainTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ChainTreeItem | undefined | null | void> =
        new vscode.EventEmitter<ChainTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ChainTreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private chainFiles: Map<string, { data: LctlChainFile; stats: ChainStats; mtime: number }> = new Map();
    private fileWatcher: vscode.FileSystemWatcher | undefined;
    private stateFileWatcher: vscode.FileSystemWatcher | undefined;
    private recordingChainPath: string | undefined;

    constructor(private context: vscode.ExtensionContext) {
        this.setupFileWatchers();
        this.checkForActiveRecording();

        context.subscriptions.push(
            vscode.workspace.onDidChangeConfiguration(e => {
                if (e.affectsConfiguration('lctl.autoRefresh')) {
                    this.disposeFileWatchers();
                    this.setupFileWatchers();
                    this.refresh();
                } else if (e.affectsConfiguration('lctl.maxEventsInTree') ||
                           e.affectsConfiguration('lctl.defaultPricing')) {
                    // Pricing feeds cached stats, so drop the cache.
                    this.refresh();
                }
            })
        );
    }

    private disposeFileWatchers(): void {
        this.fileWatcher?.dispose();
        this.stateFileWatcher?.dispose();
        this.fileWatcher = undefined;
        this.stateFileWatcher = undefined;
    }

    private setupFileWatchers(): void {
        if (!vscode.workspace.getConfiguration('lctl').get<boolean>('autoRefresh', true)) {
            return;
        }

        // Watch for .lctl.json file changes
        this.fileWatcher = vscode.workspace.createFileSystemWatcher('**/*.lctl.json');

        this.fileWatcher.onDidChange(uri => {
            this.invalidateChain(uri.fsPath);
            this._onDidChangeTreeData.fire();
        });

        this.fileWatcher.onDidCreate(_uri => {
            this._onDidChangeTreeData.fire();
        });

        this.fileWatcher.onDidDelete(uri => {
            this.chainFiles.delete(uri.fsPath);
            this._onDidChangeTreeData.fire();
        });

        // Watch for state file changes (indicates active recording)
        this.stateFileWatcher = vscode.workspace.createFileSystemWatcher('**/.lctl-state.json');

        this.stateFileWatcher.onDidChange(() => {
            this.checkForActiveRecording();
            this._onDidChangeTreeData.fire();
        });

        this.stateFileWatcher.onDidCreate(() => {
            this.checkForActiveRecording();
            this._onDidChangeTreeData.fire();
        });

        this.stateFileWatcher.onDidDelete(() => {
            this.recordingChainPath = undefined;
            this._onDidChangeTreeData.fire();
        });

    }

    private async checkForActiveRecording(): Promise<void> {
        try {
            const stateFiles = await vscode.workspace.findFiles('**/.lctl-state.json', '**/node_modules/**', 1);
            if (stateFiles.length > 0) {
                const content = await vscode.workspace.fs.readFile(stateFiles[0]);
                const state = JSON.parse(Buffer.from(content).toString('utf8'));
                this.recordingChainPath = state.chain_path;
            } else {
                this.recordingChainPath = undefined;
            }
        } catch {
            this.recordingChainPath = undefined;
        }
    }

    private invalidateChain(fsPath: string): void {
        this.chainFiles.delete(fsPath);
    }

    refresh(): void {
        this.chainFiles.clear();
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: ChainTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: ChainTreeItem): Promise<ChainTreeItem[]> {
        if (!vscode.workspace.workspaceFolders) {
            return [];
        }

        if (!element) {
            return this.getRootChains();
        }

        // Show events under chain
        if (element.chainData?.events) {
            return limitEvents(
                element.chainData.events,
                vscode.workspace.getConfiguration('lctl').get<number>('maxEventsInTree', 50)
            ).map(event =>
                new ChainTreeItem(
                    `[${event.seq}] ${event.type}`,
                    vscode.TreeItemCollapsibleState.None,
                    element.resourceUri,
                    element.chainData,
                    undefined,
                    event
                )
            );
        }

        return [];
    }

    getParent(_element: ChainTreeItem): ChainTreeItem | undefined {
        // Required for reveal() functionality
        return undefined;
    }

    private async getRootChains(): Promise<ChainTreeItem[]> {
        const chainFiles = await vscode.workspace.findFiles('**/*.lctl.json', '**/node_modules/**');
        const items: ChainTreeItem[] = [];

        for (const uri of chainFiles) {
            const result = await this.loadChainWithStats(uri);
            if (result) {
                const fileName = path.basename(uri.fsPath, '.lctl.json');
                const hasEvents = result.data.events && result.data.events.length > 0;

                items.push(
                    new ChainTreeItem(
                        fileName,
                        hasEvents
                            ? vscode.TreeItemCollapsibleState.Collapsed
                            : vscode.TreeItemCollapsibleState.None,
                        uri,
                        result.data,
                        result.stats
                    )
                );
            }
        }

        // Sort: recording first, then by event count descending
        return items.sort((a, b) => {
            if (a.stats?.isRecording && !b.stats?.isRecording) {return -1;}
            if (!a.stats?.isRecording && b.stats?.isRecording) {return 1;}
            return (b.stats?.eventCount || 0) - (a.stats?.eventCount || 0);
        });
    }

    private async loadChainWithStats(uri: vscode.Uri): Promise<{ data: LctlChainFile; stats: ChainStats } | undefined> {
        // Check cache
        const cached = this.chainFiles.get(uri.fsPath);
        if (cached) {
            const stat = await vscode.workspace.fs.stat(uri);
            if (stat.mtime === cached.mtime) {
                return { data: cached.data, stats: cached.stats };
            }
        }

        try {
            const content = await vscode.workspace.fs.readFile(uri);
            const data = JSON.parse(Buffer.from(content).toString('utf8')) as LctlChainFile;

            // Validate LCTL v4.0 schema
            if (!data.lctl || !data.chain?.id) {
                return undefined;
            }

            const stats = this.computeStats(data, uri.fsPath);
            const stat = await vscode.workspace.fs.stat(uri);

            this.chainFiles.set(uri.fsPath, { data, stats, mtime: stat.mtime });
            return { data, stats };
        } catch {
            return undefined;
        }
    }

    private computeStats(data: LctlChainFile, fsPath: string): ChainStats {
        const stats: ChainStats = {
            eventCount: data.events?.length || 0,
            errorCount: 0,
            warningCount: 0,
            totalDurationMs: 0,
            tokensIn: 0,
            tokensOut: 0,
            estimatedCost: 0,
            isRecording: this.isRecording(fsPath),
            agents: new Set(),
            lastModified: new Date(),
        };

        if (!data.events) {
            return stats;
        }

        for (const event of data.events) {
            // Collect agents
            if (event.agent) {
                stats.agents.add(event.agent);
            }

            // Count errors
            if (event.type === 'error') {
                stats.errorCount++;
            }

            // Accumulate metrics from step_end events
            if (event.type === 'step_end' && event.data) {
                stats.totalDurationMs += Number(event.data.duration_ms) || 0;
            }
        }

        const tokens = chainTokens(data.events);
        stats.tokensIn = tokens.input;
        stats.tokensOut = tokens.output;
        stats.estimatedCost = chainCost(tokens, pricingFor(configuredModel()));

        return stats;
    }

    private isRecording(fsPath: string): boolean {
        if (!this.recordingChainPath) {
            return false;
        }
        return fsPath.endsWith(this.recordingChainPath) ||
               this.recordingChainPath.endsWith(path.basename(fsPath));
    }

    async getChainData(uri: vscode.Uri): Promise<LctlChainFile | undefined> {
        const result = await this.loadChainWithStats(uri);
        return result?.data;
    }

    findChainUri(chainId: string): vscode.Uri | undefined {
        for (const [fsPath, cached] of this.chainFiles) {
            if (cached.data.chain?.id === chainId) {
                return vscode.Uri.file(fsPath);
            }
        }
        return undefined;
    }

    dispose(): void {
        this.disposeFileWatchers();
    }
}
