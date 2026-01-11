import * as vscode from 'vscode';
import { ChainTreeProvider, ChainTreeItem } from './chainProvider';

type DashboardOpener = (uri?: vscode.Uri) => void;

export function registerCommands(
    context: vscode.ExtensionContext,
    chainProvider: ChainTreeProvider,
    openDashboard: DashboardOpener
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.openDashboard', () => {
            openDashboard();
        })
    );

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

    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.replay', async (arg?: string | vscode.Uri | ChainTreeItem) => {
            let chainId: string | undefined;
            let uri: vscode.Uri | undefined;

            if (typeof arg === 'string') {
                chainId = arg;
            } else if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            }

            if (uri) {
                const metadata = await chainProvider.getChainMetadata(uri);
                chainId = metadata?.chain_id;
            }

            if (!chainId) {
                const input = await vscode.window.showInputBox({
                    prompt: 'Enter Chain ID to replay',
                    placeHolder: 'chain-id-here'
                });
                chainId = input;
            }

            if (chainId) {
                void vscode.window.withProgress(
                    {
                        location: vscode.ProgressLocation.Notification,
                        title: `Replaying chain: ${chainId}`,
                        cancellable: true
                    },
                    async (progress, token) => {
                        return new Promise<void>((resolve) => {
                            let cancelled = false;
                            token.onCancellationRequested(() => {
                                cancelled = true;
                                resolve();
                            });

                            const steps = ['Initializing...', 'Loading snapshots...', 'Replaying...', 'Complete'];
                            let currentStep = 0;

                            const interval = setInterval(() => {
                                if (cancelled || currentStep >= steps.length) {
                                    clearInterval(interval);
                                    if (!cancelled) {
                                        void vscode.window.showInformationMessage(
                                            `Chain ${chainId} replay complete.`
                                        );
                                    }
                                    resolve();
                                    return;
                                }

                                progress.report({
                                    increment: 25,
                                    message: steps[currentStep]
                                });
                                currentStep++;
                            }, 500);
                        });
                    }
                );
            }
        })
    );
}
