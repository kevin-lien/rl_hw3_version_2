param(
    [ValidateSet('online','offline')][string]$Mode = 'online',
    [string]$Entity = '',
    [int]$Epochs = 5,
    [string]$SavePath = 'models/model_astro_party'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:WANDB_DATA_DIR = Join-Path $PSScriptRoot '.wandb-data'
$env:WANDB_CACHE_DIR = Join-Path $PSScriptRoot '.wandb-cache'
$pythonExe = Join-Path $PSScriptRoot '../.venv311/Scripts/python.exe'
$trainArgs = @('train_astro_party.py', '--name', 'defense-ppo-h60-final',
    '--wandb-mode', $Mode, '--features', 'tactical', '--controller', 'defense', '--select-last',
    '--near', '500', '--far', '700', '--horizon', '60',
    '--bc-steps', '10000', '--bc-epochs', '120', '--epochs', "$Epochs",
    '--steps-per-epoch', '8192', '--eval-episodes', '50',
    '--learning-rate', '0.000001', '--save-path', $SavePath)
if ($Entity) { $trainArgs += @('--entity', $Entity) }
& $pythonExe @trainArgs
if ($LASTEXITCODE -ne 0) { throw 'Training did not complete successfully; inspect the preceding error.' }
