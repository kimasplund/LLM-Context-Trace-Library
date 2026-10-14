import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { LctlChainFile, configuredModel } from './chainProvider';
import { pricingFor } from './chainStats';

export class LctlDashboardPanel {
    public static currentPanel: LctlDashboardPanel | undefined;
    public static readonly viewType = 'lctlDashboard';

    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _chainUri: vscode.Uri | undefined;
    private _chainData: LctlChainFile | undefined;
    private _disposables: vscode.Disposable[] = [];
    private _fileWatcher: vscode.FileSystemWatcher | undefined;
    private _stateWatcher: vscode.FileSystemWatcher | undefined;
    private _isRecording: boolean = false;

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
                    vscode.Uri.joinPath(extensionUri, 'out', 'webview'),
                    vscode.Uri.joinPath(extensionUri, 'media')
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

        this._update();
        this._setupStateWatcher();

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

        vscode.workspace.onDidChangeConfiguration(
            e => {
                if (this._chainUri && e.affectsConfiguration('lctl')) {
                    // autoRefresh toggles the watcher; defaultPricing changes the cost estimate.
                    void this.loadChain(this._chainUri);
                }
            },
            null,
            this._disposables
        );

        if (chainUri) {
            this.loadChain(chainUri);
        }
    }

    public async loadChain(uri: vscode.Uri): Promise<void> {
        this._chainUri = uri;
        this._setupFileWatcher(uri);
        await this._loadAndDisplayChain();
    }

    private _setupFileWatcher(uri: vscode.Uri): void {
        this._fileWatcher?.dispose();
        this._fileWatcher = undefined;

        if (!vscode.workspace.getConfiguration('lctl').get<boolean>('autoRefresh', true)) {
            return;
        }

        const pattern = new vscode.RelativePattern(
            vscode.Uri.joinPath(uri, '..'),
            '*.lctl.json'
        );
        this._fileWatcher = vscode.workspace.createFileSystemWatcher(pattern);

        this._fileWatcher.onDidChange(changedUri => {
            if (changedUri.fsPath === uri.fsPath) {
                this._loadAndDisplayChain();
            }
        });

        this._disposables.push(this._fileWatcher);
    }

    private _setupStateWatcher(): void {
        this._stateWatcher = vscode.workspace.createFileSystemWatcher('**/.lctl-state.json');

        const checkRecording = async () => {
            try {
                const stateFiles = await vscode.workspace.findFiles('**/.lctl-state.json', '**/node_modules/**', 1);
                this._isRecording = stateFiles.length > 0;
                this._panel.webview.postMessage({
                    type: 'recordingUpdate',
                    isRecording: this._isRecording
                });
            } catch {
                this._isRecording = false;
            }
        };

        this._stateWatcher.onDidCreate(checkRecording);
        this._stateWatcher.onDidDelete(checkRecording);
        this._stateWatcher.onDidChange(checkRecording);
        this._disposables.push(this._stateWatcher);

        // Initial check
        checkRecording();
    }

    private async _loadAndDisplayChain(): Promise<void> {
        if (!this._chainUri) {return;}

        try {
            const content = await vscode.workspace.fs.readFile(this._chainUri);
            this._chainData = JSON.parse(Buffer.from(content).toString('utf8'));

            this._panel.webview.postMessage({
                type: 'chainData',
                data: this._chainData,
                path: this._chainUri.fsPath,
                isRecording: this._isRecording,
                pricing: pricingFor(configuredModel())
            });
        } catch (err) {
            this._panel.webview.postMessage({
                type: 'error',
                message: `Failed to load chain: ${err}`
            });
        }
    }

    public dispose(): void {
        LctlDashboardPanel.currentPanel = undefined;

        this._fileWatcher?.dispose();
        this._stateWatcher?.dispose();
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
                    await this._loadAndDisplayChain();
                }
                break;

            case 'openFile':
                if (message.path) {
                    const uri = vscode.Uri.file(message.path);
                    const doc = await vscode.workspace.openTextDocument(uri);
                    const line = message.line ? message.line - 1 : 0;
                    await vscode.window.showTextDocument(doc, {
                        selection: new vscode.Range(line, 0, line, 0)
                    });
                }
                break;

            case 'replay':
                if (this._chainUri) {
                    await vscode.commands.executeCommand('lctl.replay', this._chainUri);
                }
                break;

            case 'exportHtml':
                if (this._chainUri) {
                    await vscode.commands.executeCommand('lctl.exportHtml', this._chainUri);
                }
                break;

            case 'showStats':
                if (this._chainUri) {
                    await vscode.commands.executeCommand('lctl.showStats', this._chainUri);
                }
                break;

            case 'requestData':
                if (this._chainUri) {
                    await this._loadAndDisplayChain();
                }
                break;

            case 'requestCompare':
                await this._handleCompareRequest();
                break;
        }
    }

    private async _handleCompareRequest(): Promise<void> {
        // Get all chain files
        const files = await vscode.workspace.findFiles('**/*.lctl.json', '**/node_modules/**');

        if (files.length < 2) {
            void vscode.window.showWarningMessage('Need at least 2 chain files to compare.');
            return;
        }

        // Filter out current chain
        const otherFiles = this._chainUri
            ? files.filter(f => f.fsPath !== this._chainUri!.fsPath)
            : files;

        // Let user pick
        const items = otherFiles.map((file) => ({
            label: path.basename(file.fsPath, '.lctl.json'),
            description: vscode.workspace.asRelativePath(file),
            uri: file
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a chain to compare with'
        });

        if (!selected) {return;}

        // Load the comparison chain
        try {
            const content = await vscode.workspace.fs.readFile(selected.uri);
            const data = JSON.parse(Buffer.from(content).toString('utf8'));

            this._panel.webview.postMessage({
                type: 'compareChainData',
                data,
                path: selected.uri.fsPath
            });
        } catch (err) {
            void vscode.window.showErrorMessage(`Failed to load comparison chain: ${err}`);
        }
    }

    private _getHtmlForWebview(): string {
        const webview = this._panel.webview;
        const webviewPath = vscode.Uri.joinPath(this._extensionUri, 'out', 'webview');

        // Check if React build exists
        const indexPath = path.join(webviewPath.fsPath, 'index.html');
        if (fs.existsSync(indexPath)) {
            return this._getReactWebviewHtml(webview, webviewPath);
        }

        // Fallback to inline HTML if React build doesn't exist
        return this._getFallbackHtml();
    }

    private _getReactWebviewHtml(webview: vscode.Webview, webviewPath: vscode.Uri): string {
        // Read the built index.html
        const indexPath = path.join(webviewPath.fsPath, 'index.html');
        let html = fs.readFileSync(indexPath, 'utf8');

        // Get URIs for assets
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(webviewPath, 'assets', 'index.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(webviewPath, 'assets', 'style.css'));

        // Generate nonce for CSP
        const nonce = getNonce();

        // Update CSP and asset paths
        html = html
            // Update CSP to allow our scripts and styles
            .replace(
                /<meta http-equiv="Content-Security-Policy"[^>]*>/,
                `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data:; font-src ${webview.cspSource} data:;">`
            )
            // Replace script tag (Vite adds crossorigin attribute)
            .replace(
                /<script type="module"[^>]*src="[^"]*"[^>]*><\/script>/,
                `<script type="module" nonce="${nonce}" src="${scriptUri}"></script>`
            )
            // Replace stylesheet link (Vite adds crossorigin attribute)
            .replace(
                /<link rel="stylesheet"[^>]*href="[^"]*"[^>]*>/,
                `<link rel="stylesheet" href="${styleUri}">`
            );

        return html;
    }

    private _getFallbackHtml(): string {
        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCTL Dashboard</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background: var(--vscode-editor-background);
            color: var(--vscode-foreground);
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
        }
        .message {
            text-align: center;
            max-width: 400px;
        }
        .icon { font-size: 48px; margin-bottom: 16px; }
        h2 { margin-bottom: 8px; }
        p { opacity: 0.7; margin-bottom: 16px; }
        .code {
            background: var(--vscode-textCodeBlock-background);
            padding: 8px 12px;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="message">
        <div class="icon">🔧</div>
        <h2>React Dashboard Not Built</h2>
        <p>The React webview hasn't been built yet. Run the following command in the webview-ui directory:</p>
        <div class="code">npm run build</div>
    </div>
    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        vscode.postMessage({ type: 'ready' });
    </script>
</body>
</html>`;
    }
}

interface WebviewMessage {
    type: 'ready' | 'openFile' | 'replay' | 'exportHtml' | 'showStats' | 'requestData' | 'requestCompare';
    path?: string;
    line?: number;
}

function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
