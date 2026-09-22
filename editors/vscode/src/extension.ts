// The extension is a thin client: it starts `linnet lsp --stdio` and lets the
// server provide every language feature. No language logic lives here.

import * as vscode from "vscode";
import { LanguageClient, LanguageClientOptions, ServerOptions } from "vscode-languageclient/node";

let client: LanguageClient | undefined;

function serverOptions(): ServerOptions {
    const configuration = vscode.workspace.getConfiguration("linnet");
    const command = configuration.get<string>("path", "linnet");
    const stdRoot = configuration.get<string>("stdRoot", "");
    const args = ["lsp", "--stdio"];
    if (stdRoot !== "") {
        args.push("--std", stdRoot);
    }
    return { command, args };
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: "file", language: "linnet" }],
    };
    client = new LanguageClient("linnet", "Linnet", serverOptions(), clientOptions);
    context.subscriptions.push(
        vscode.commands.registerCommand("linnet.restartServer", async () => {
            await client?.restart();
        }),
    );
    try {
        await client.start();
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        void vscode.window.showWarningMessage(
            `Linnet: could not start the language server (${message}). ` +
                "Set `linnet.path` to the linnet executable.",
        );
    }
}

export async function deactivate(): Promise<void> {
    await client?.stop();
    client = undefined;
}
