import * as vscode from 'vscode';
import * as path from 'path';

export interface LctlChainMetadata {
    version: string;
    chain_id: string;
    created: string;
    snapshots?: LctlSnapshot[];
}

export interface LctlSnapshot {
    snapshot_id: string;
    timestamp: string;
    description?: string;
}

export class ChainTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly resourceUri?: vscode.Uri,
        public readonly chainData?: LctlChainMetadata,
        public readonly snapshot?: LctlSnapshot
    ) {
        super(label, collapsibleState);

        if (resourceUri && !snapshot) {
            this.contextValue = 'chainFile';
            this.iconPath = new vscode.ThemeIcon('git-branch');
            this.tooltip = `Chain: ${chainData?.chain_id ?? label}`;
            this.description = chainData?.version ?? '';
            this.command = {
                command: 'lctl.loadChain',
                title: 'Load Chain',
                arguments: [resourceUri]
            };
        } else if (snapshot) {
            this.contextValue = 'snapshot';
            this.iconPath = new vscode.ThemeIcon('git-commit');
            this.tooltip = snapshot.description ?? snapshot.snapshot_id;
            this.description = new Date(snapshot.timestamp).toLocaleString();
        }
    }
}

export class ChainTreeProvider implements vscode.TreeDataProvider<ChainTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ChainTreeItem | undefined | null | void> =
        new vscode.EventEmitter<ChainTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ChainTreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private chainFiles: Map<string, LctlChainMetadata> = new Map();

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

        if (element.chainData?.snapshots) {
            return element.chainData.snapshots.map(
                (snapshot) =>
                    new ChainTreeItem(
                        snapshot.snapshot_id,
                        vscode.TreeItemCollapsibleState.None,
                        element.resourceUri,
                        element.chainData,
                        snapshot
                    )
            );
        }

        return [];
    }

    private async getRootChains(): Promise<ChainTreeItem[]> {
        const chainFiles = await vscode.workspace.findFiles('**/*.lctl.json', '**/node_modules/**');
        const items: ChainTreeItem[] = [];

        for (const uri of chainFiles) {
            const metadata = await this.loadChainMetadata(uri);
            const fileName = path.basename(uri.fsPath, '.lctl.json');
            const hasSnapshots = metadata?.snapshots && metadata.snapshots.length > 0;

            items.push(
                new ChainTreeItem(
                    fileName,
                    hasSnapshots
                        ? vscode.TreeItemCollapsibleState.Collapsed
                        : vscode.TreeItemCollapsibleState.None,
                    uri,
                    metadata
                )
            );
        }

        return items.sort((a, b) => a.label.localeCompare(b.label));
    }

    private async loadChainMetadata(uri: vscode.Uri): Promise<LctlChainMetadata | undefined> {
        const cached = this.chainFiles.get(uri.fsPath);
        if (cached) {
            return cached;
        }

        try {
            const content = await vscode.workspace.fs.readFile(uri);
            const data = JSON.parse(Buffer.from(content).toString('utf8')) as LctlChainMetadata;
            this.chainFiles.set(uri.fsPath, data);
            return data;
        } catch {
            return undefined;
        }
    }

    async getChainMetadata(uri: vscode.Uri): Promise<LctlChainMetadata | undefined> {
        return this.loadChainMetadata(uri);
    }
}
