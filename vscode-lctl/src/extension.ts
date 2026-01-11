import * as vscode from 'vscode';
import { ChainTreeProvider } from './chainProvider';
import { LctlDashboardPanel } from './webviewPanel';
import { registerCommands } from './commands';

let chainProvider: ChainTreeProvider;

export function activate(context: vscode.ExtensionContext): void {
    chainProvider = new ChainTreeProvider();

    const treeView = vscode.window.createTreeView('lctlExplorer', {
        treeDataProvider: chainProvider,
        showCollapseAll: true
    });

    context.subscriptions.push(treeView);

    registerCommands(context, chainProvider, (uri?: vscode.Uri) => {
        LctlDashboardPanel.createOrShow(context.extensionUri, uri);
    });

    const watcher = vscode.workspace.createFileSystemWatcher('**/*.lctl.json');
    watcher.onDidCreate(() => chainProvider.refresh());
    watcher.onDidDelete(() => chainProvider.refresh());
    watcher.onDidChange(() => chainProvider.refresh());
    context.subscriptions.push(watcher);
}

export function deactivate(): void {
    // Cleanup handled by disposables
}
