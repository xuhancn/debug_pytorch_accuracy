# Setup VSCode debug cfg

For `.vscode\launch.json`:
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "--accuracy",
                "-d",
                "xpu",
                "-n1",
                "--backend=inductor",
                "--cold-start-latency",
                "--training",
                "--bfloat16",
                "--only",
                "OPTForCausalLM"
            ]
        }
    ]
}
```