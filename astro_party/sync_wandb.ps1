param(
    [string]$Entity = '',
    [string]$RunPath = 'wandb/offline-run-20261101_121443-6o6gk5el'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:WANDB_DATA_DIR = Join-Path $PSScriptRoot '.wandb-data'
$env:WANDB_CACHE_DIR = Join-Path $PSScriptRoot '.wandb-cache'
$wandbExe = Join-Path $PSScriptRoot '../.venv311/Scripts/wandb.exe'
$syncArgs = @('sync', '--yes', '--project', 'astro-party-hw3', $RunPath)
if ($Entity) { $syncArgs += @('--entity', $Entity) }
& $wandbExe @syncArgs
if ($LASTEXITCODE -ne 0) { throw 'Sync failed. Run the same environment wandb login and retry.' }
