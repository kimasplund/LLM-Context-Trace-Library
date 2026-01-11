import * as vscode from 'vscode';

export class LctlDashboardPanel {
    public static currentPanel: LctlDashboardPanel | undefined;
    public static readonly viewType = 'lctlDashboard';

    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _chainUri: vscode.Uri | undefined;
    private _disposables: vscode.Disposable[] = [];

    public static createOrShow(extensionUri: vscode.Uri, chainUri?: vscode.Uri): void {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (LctlDashboardPanel.currentPanel) {
            LctlDashboardPanel.currentPanel._panel.reveal(column);
            if (chainUri) {
                LctlDashboardPanel.currentPanel.loadChain(chainUri);
            }
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            LctlDashboardPanel.viewType,
            'LCTL Dashboard',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(extensionUri, 'media'),
                    vscode.Uri.joinPath(extensionUri, 'out')
                ]
            }
        );

        LctlDashboardPanel.currentPanel = new LctlDashboardPanel(panel, extensionUri, chainUri);
    }

    public static revive(panel: vscode.WebviewPanel, extensionUri: vscode.Uri): void {
        LctlDashboardPanel.currentPanel = new LctlDashboardPanel(panel, extensionUri);
    }

    private constructor(
        panel: vscode.WebviewPanel,
        extensionUri: vscode.Uri,
        chainUri?: vscode.Uri
    ) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._chainUri = chainUri;

        this._update();

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.onDidChangeViewState(
            () => {
                if (this._panel.visible) {
                    this._update();
                }
            },
            null,
            this._disposables
        );

        this._panel.webview.onDidReceiveMessage(
            (message: WebviewMessage) => this._handleMessage(message),
            null,
            this._disposables
        );
    }

    public loadChain(uri: vscode.Uri): void {
        this._chainUri = uri;
        this._panel.webview.postMessage({
            type: 'loadChain',
            uri: uri.fsPath
        });
    }

    public dispose(): void {
        LctlDashboardPanel.currentPanel = undefined;

        this._panel.dispose();

        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }

    private _update(): void {
        this._panel.title = 'LCTL Dashboard';
        this._panel.webview.html = this._getHtmlForWebview();
    }

    private async _handleMessage(message: WebviewMessage): Promise<void> {
        switch (message.type) {
            case 'ready':
                if (this._chainUri) {
                    this.loadChain(this._chainUri);
                }
                break;
            case 'openFile':
                if (message.path) {
                    const uri = vscode.Uri.file(message.path);
                    await vscode.window.showTextDocument(uri);
                }
                break;
            case 'replay':
                if (message.chainId) {
                    await vscode.commands.executeCommand('lctl.replay', message.chainId);
                }
                break;
            case 'showMessage':
                if (message.text) {
                    void vscode.window.showInformationMessage(message.text);
                }
                break;
        }
    }

    private _getHtmlForWebview(): string {
        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCTL Dashboard</title>
    <style>
        :root {
            --container-padding: 20px;
            --input-padding-vertical: 6px;
            --input-padding-horizontal: 10px;
        }
        body {
            padding: var(--container-padding);
            color: var(--vscode-foreground);
            font-size: var(--vscode-font-size);
            font-weight: var(--vscode-font-weight);
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
        }
        .header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--vscode-widget-border);
        }
        .header h1 {
            margin: 0;
            font-size: 1.5em;
            font-weight: 600;
        }
        .version-badge {
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .chain-info {
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .chain-info h2 {
            margin: 0 0 12px 0;
            font-size: 1.1em;
        }
        .info-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 8px 16px;
        }
        .info-label {
            color: var(--vscode-descriptionForeground);
        }
        .info-value {
            font-family: var(--vscode-editor-font-family);
        }
        .actions {
            display: flex;
            gap: 8px;
            margin-top: 20px;
        }
        button {
            border: none;
            padding: var(--input-padding-vertical) var(--input-padding-horizontal);
            text-align: center;
            outline: 1px solid transparent;
            outline-offset: 2px;
            color: var(--vscode-button-foreground);
            background: var(--vscode-button-background);
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        button:focus {
            outline-color: var(--vscode-focusBorder);
        }
        button.secondary {
            color: var(--vscode-button-secondaryForeground);
            background: var(--vscode-button-secondaryBackground);
        }
        button.secondary:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }
        .empty-state {
            text-align: center;
            padding: 48px 24px;
            color: var(--vscode-descriptionForeground);
        }
        .empty-state h2 {
            margin-bottom: 8px;
        }
        .timeline {
            margin-top: 24px;
        }
        .timeline h2 {
            margin-bottom: 16px;
        }
        .timeline-item {
            display: flex;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid var(--vscode-widget-border);
        }
        .timeline-marker {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--vscode-button-background);
            margin-top: 4px;
        }
        .timeline-content {
            flex: 1;
        }
        .timeline-title {
            font-weight: 500;
        }
        .timeline-time {
            font-size: 0.85em;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>LCTL Time-Travel Debugger</h1>
        <span class="version-badge">v4.0</span>
    </div>

    <div id="empty-state" class="empty-state">
        <h2>No Chain Loaded</h2>
        <p>Select a chain from the LCTL Explorer to view its details.</p>
    </div>

    <div id="chain-content" style="display: none;">
        <div class="chain-info">
            <h2>Chain Information</h2>
            <div class="info-grid">
                <span class="info-label">Chain ID:</span>
                <span class="info-value" id="chain-id">-</span>
                <span class="info-label">Version:</span>
                <span class="info-value" id="chain-version">-</span>
                <span class="info-label">Created:</span>
                <span class="info-value" id="chain-created">-</span>
                <span class="info-label">File:</span>
                <span class="info-value" id="chain-file">-</span>
            </div>
            <div class="actions">
                <button id="btn-replay">Replay Chain</button>
                <button id="btn-open" class="secondary">Open File</button>
            </div>
        </div>

        <div class="timeline" id="timeline">
            <h2>Snapshots</h2>
            <div id="timeline-items"></div>
        </div>
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        let currentChainPath = null;

        function showEmptyState() {
            document.getElementById('empty-state').style.display = 'block';
            document.getElementById('chain-content').style.display = 'none';
        }

        function showChainContent() {
            document.getElementById('empty-state').style.display = 'none';
            document.getElementById('chain-content').style.display = 'block';
        }

        async function loadChainData(path) {
            currentChainPath = path;
            try {
                const response = await fetch('vscode-resource:' + path);
                const data = await response.json();
                displayChain(data, path);
            } catch (err) {
                showEmptyState();
            }
        }

        function displayChain(data, path) {
            document.getElementById('chain-id').textContent = data.chain_id || '-';
            document.getElementById('chain-version').textContent = data.version || '-';
            document.getElementById('chain-created').textContent = data.created
                ? new Date(data.created).toLocaleString()
                : '-';
            document.getElementById('chain-file').textContent = path.split('/').pop();

            const timelineItems = document.getElementById('timeline-items');
            timelineItems.innerHTML = '';

            if (data.snapshots && data.snapshots.length > 0) {
                data.snapshots.forEach(snapshot => {
                    const item = document.createElement('div');
                    item.className = 'timeline-item';
                    item.innerHTML = \`
                        <div class="timeline-marker"></div>
                        <div class="timeline-content">
                            <div class="timeline-title">\${snapshot.snapshot_id}</div>
                            <div class="timeline-time">\${new Date(snapshot.timestamp).toLocaleString()}</div>
                            \${snapshot.description ? \`<div>\${snapshot.description}</div>\` : ''}
                        </div>
                    \`;
                    timelineItems.appendChild(item);
                });
            } else {
                timelineItems.innerHTML = '<p style="color: var(--vscode-descriptionForeground);">No snapshots in this chain.</p>';
            }

            showChainContent();
        }

        document.getElementById('btn-replay').addEventListener('click', () => {
            const chainId = document.getElementById('chain-id').textContent;
            vscode.postMessage({ type: 'replay', chainId });
        });

        document.getElementById('btn-open').addEventListener('click', () => {
            if (currentChainPath) {
                vscode.postMessage({ type: 'openFile', path: currentChainPath });
            }
        });

        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'loadChain':
                    loadChainData(message.uri);
                    break;
            }
        });

        vscode.postMessage({ type: 'ready' });
    </script>
</body>
</html>`;
    }
}

interface WebviewMessage {
    type: 'ready' | 'openFile' | 'replay' | 'showMessage';
    path?: string;
    chainId?: string;
    text?: string;
}

function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
