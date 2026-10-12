#Requires -Version 5.1
<#
.SYNOPSIS
    Builds Whisp for Windows: the app folder (dist\Whisp) and, with -Installer, the setup program.

.DESCRIPTION
    Runs PyInstaller from a build environment (build\env) that uv makes from
    uv.lock every time, so the app holds exactly the locked versions and nothing
    else that happens to be installed in .venv or on this PC. Nothing is
    installed globally. Needs uv: https://docs.astral.sh/uv/

    The app is built into dist\Whisp\ (start dist\Whisp\Whisp.exe). The script
    then checks that Whisp.exe starts and reports the right version.

    With -Installer, Inno Setup 6 then packs the app into
    dist\Whisp-Setup-<version>.exe. Inno Setup is not part of the Python
    environment; install it from https://jrsoftware.org/isinfo.php (a per-user
    install is fine) or pass its location with -InnoSetup.

.PARAMETER Installer
    Also build the installer with Inno Setup 6.

.PARAMETER InnoSetup
    Full path to Inno Setup's compiler, ISCC.exe, if it isn't on PATH or in
    Inno Setup's usual install folders.

.EXAMPLE
    .\scripts\build.ps1

.EXAMPLE
    .\scripts\build.ps1 -Installer
#>
[CmdletBinding()]
param(
    [switch]$Installer,
    [string]$InnoSetup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$spec = Join-Path $root 'packaging\whisp.spec'
$script = Join-Path $root 'packaging\whisp.iss'
$distDir = Join-Path $root 'dist'
$workDir = Join-Path $root 'build'
$buildEnv = Join-Path $workDir 'env'
$python = Join-Path $buildEnv 'Scripts\python.exe'
$appDir = Join-Path $distDir 'Whisp'
$appExe = Join-Path $appDir 'Whisp.exe'

function Get-WhispVersion {
    $init = Join-Path $root 'src\whisp\__init__.py'
    $match = Select-String -LiteralPath $init -Pattern '^__version__\s*=\s*"(\d+\.\d+\.\d+)"' | Select-Object -First 1
    if (-not $match) {
        throw "Couldn't read __version__ from $init."
    }
    return $match.Matches[0].Groups[1].Value
}

function Get-FolderSize([string]$Path) {
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    return [int64]$sum
}

function Format-Size([int64]$Bytes) {
    return '{0:N1} MB' -f ($Bytes / 1MB)
}

function Invoke-Tool {
    # PyInstaller and Inno Setup report progress on stderr. Windows PowerShell 5.1 turns those lines
    # into terminating errors when the output is redirected, so run them with 'Continue' here and
    # judge success by $LASTEXITCODE afterwards.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    $ErrorActionPreference = 'Continue'
    & $FilePath @ArgumentList
}

function Find-Uv {
    $onPath = Get-Command 'uv.exe' -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    # Where uv's standalone installer puts it.
    $installed = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $installed -PathType Leaf) {
        return $installed
    }
    throw 'uv was not found. Install it from https://docs.astral.sh/uv/ and run this script again.'
}

function Find-InnoSetupCompiler {
    if ($InnoSetup) {
        if (-not (Test-Path -LiteralPath $InnoSetup -PathType Leaf)) {
            throw "ISCC.exe not found at $InnoSetup."
        }
        return $InnoSetup
    }
    $onPath = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw 'Inno Setup 6 was not found. Install it from https://jrsoftware.org/isinfo.php or pass -InnoSetup <path to ISCC.exe>.'
}

$version = Get-WhispVersion
Write-Host "Building Whisp $version" -ForegroundColor Cyan

# The build environment: the app's dependencies and the dev tools (PyInstaller) at exactly the versions
# in uv.lock. --locked fails if pyproject.toml changed without updating the lock; uv also removes anything
# the lock doesn't list. Whisp's own code comes from src\ (see the spec), so the project isn't installed.
$uv = Find-Uv
Write-Host "Syncing the build environment with uv.lock" -ForegroundColor Cyan
$env:UV_PROJECT_ENVIRONMENT = $buildEnv
try {
    Invoke-Tool $uv @('sync', '--locked', '--no-install-project', '--project', $root)
    if ($LASTEXITCODE -ne 0) {
        throw "uv couldn't set up the build environment (exit code $LASTEXITCODE)."
    }
}
finally {
    Remove-Item Env:UV_PROJECT_ENVIRONMENT
}
# find_spec prints nothing, so a missing package can't trip PowerShell's error handling.
& $python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is missing from the build environment. It belongs in the dev dependency group in pyproject.toml."
}

# PyInstaller looks for the DLLs that the app's libraries need on PATH. Limit PATH to
# Windows' own folders while it runs, so nothing from other programs on this PC (a CUDA
# toolkit, a stray OpenSSL, ...) can end up inside Whisp by accident.
$savedPath = $env:PATH
try {
    $env:PATH = @(
        (Join-Path $env:SystemRoot 'System32'),
        $env:SystemRoot,
        (Join-Path $env:SystemRoot 'System32\Wbem')
    ) -join ';'
    Invoke-Tool $python @('-m', 'PyInstaller', '--noconfirm', '--clean', '--distpath', $distDir, '--workpath', $workDir, $spec)
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed (exit code $LASTEXITCODE)."
    }
}
finally {
    $env:PATH = $savedPath
}

# Smoke test: the frozen app starts, imports its entry point and knows its version.
# Piping the output makes PowerShell wait for the (windowed) Whisp.exe to finish.
$reported = (& $appExe --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $reported -ne "Whisp $version") {
    throw "dist\Whisp\Whisp.exe --version printed '$reported' (exit code $LASTEXITCODE); expected 'Whisp $version'."
}
Write-Host "App: $appDir ($(Format-Size (Get-FolderSize $appDir)))" -ForegroundColor Green

if ($Installer) {
    $iscc = Find-InnoSetupCompiler
    Write-Host "Building the installer with $iscc" -ForegroundColor Cyan
    Invoke-Tool $iscc @('/Qp', $script)
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed (exit code $LASTEXITCODE)."
    }
    $setup = Join-Path $distDir "Whisp-Setup-$version.exe"
    if (-not (Test-Path -LiteralPath $setup -PathType Leaf)) {
        throw "Inno Setup finished but $setup is missing."
    }
    Write-Host "Installer: $setup ($(Format-Size (Get-Item -LiteralPath $setup).Length))" -ForegroundColor Green
}
