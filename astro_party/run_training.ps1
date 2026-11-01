param(
    [ValidateSet('auto','online','offline')][string]$Mode = 'auto',
    [string]$Entity = '',
    [int]$Epochs = 5,
    [string]$SavePath = 'models/model_astro_party'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:WANDB_DATA_DIR = Join-Path $PSScriptRoot '.wandb-data'
$env:WANDB_CACHE_DIR = Join-Path $PSScriptRoot '.wandb-cache'
$pythonExe = Join-Path $PSScriptRoot '../.venv311/Scripts/python.exe'
# All learning defaults live in the Python file, including the TA's bare command.
$trainArgs = @('train_astro_party.py', '--wandb-mode', $Mode,
    '--epochs', "$Epochs", '--save-path', $SavePath)
if ($Entity) { $trainArgs += @('--entity', $Entity) }
& $pythonExe @trainArgs
if ($LASTEXITCODE -ne 0) { throw 'Training did not complete successfully; inspect the preceding error.' }
