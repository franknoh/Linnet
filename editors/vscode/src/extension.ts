// The extension is a thin client: it starts `linnet lsp --stdio` and lets the
// server provide every language feature. No language logic lives here.

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { LanguageClient, LanguageClientOptions, ServerOptions } from "vscode-languageclient/node";

let client: LanguageClient | undefined;

// The compiler: `linnet.path` when set, else the one a platform build of the
// extension bundles under `server/` (with its standard library beside it),
// else `linnet` on PATH.
function compiler(context: vscode.ExtensionContext): string {
    const configured = vscode.workspace.getConfiguration("linnet").get<string>("path", "");
    if (configured !== "") {
        return configured;
    }
    const name = process.platform === "win32" ? "linnet.exe" : "linnet";
    const bundled = path.join(context.extensionPath, "server", "bin", name);
    if (!fs.existsSync(bundled)) {
        return "linnet";
    }
    if (process.platform !== "win32") {
        // An unpacked extension may have lost the executable bit.
        try {
            fs.accessSync(bundled, fs.constants.X_OK);
        } catch {
            fs.chmodSync(bundled, 0o755);
        }
    }
    return bundled;
}

function serverOptions(context: vscode.ExtensionContext): ServerOptions {
    const configuration = vscode.workspace.getConfiguration("linnet");
    const command = compiler(context);
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
    client = new LanguageClient("linnet", "Linnet", serverOptions(context), clientOptions);
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
                "Set `linnet.path` to the linnet executable, or install it with " +
                "`pip install linnet-lang`.",
        );
    }
}

export async function deactivate(): Promise<void> {
    await client?.stop();
    client = undefined;
}
