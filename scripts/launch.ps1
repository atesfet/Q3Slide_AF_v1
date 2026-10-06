$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$appDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $appDir
New-Item -ItemType Directory -Force -Path "$appDir\.runtime\downloads", "$appDir\.runtime\cache" | Out-Null
Write-Host "`nQ3Slide AF - preparing your local workspace`n"
$candidates = @($env:CONDA_EXE, "$appDir\.runtime\miniforge\Scripts\conda.exe")
$command = Get-Command conda.exe -ErrorAction SilentlyContinue
if ($command) { $candidates += $command.Source }
foreach ($base in @("$env:USERPROFILE\miniforge3", "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniforge3", "$env:LOCALAPPDATA\miniconda3", "$env:ProgramData\miniconda3", "$env:ProgramData\anaconda3")) {
    $candidates += "$base\Scripts\conda.exe"
}
if ($env:Q3SLIDE_LOCAL_CONDA -eq '1') { $candidates = @("$appDir\.runtime\miniforge\Scripts\conda.exe") }
$condaExe = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) -and $_.EndsWith('.exe') } | Select-Object -First 1
if (-not $condaExe) {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Q3Slide AF requires 64-bit Windows.' }
    $version = '26.7.2-0'
    $filename = "Miniforge3-$version-Windows-x86_64.exe"
    $url = "https://github.com/conda-forge/miniforge/releases/download/$version/$filename"
    $installer = "$appDir\.runtime\downloads\$filename"
    Write-Host 'Conda was not found. Downloading Miniforge into this application folder…'
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
    Invoke-WebRequest -UseBasicParsing -Uri "$url.sha256" -OutFile "$installer.sha256"
    $expected = ((Get-Content -LiteralPath "$installer.sha256" -Raw).Trim() -split '\s+')[0]
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash
    if ($expected -notmatch '^[a-fA-F0-9]{64}$' -or $actual -ne $expected) { throw 'Installer checksum verification failed. Please launch again.' }
    $prefix = "$appDir\.runtime\miniforge"
    # /D must be the last NSIS argument. Do not quote the /D value.
    $process = Start-Process -FilePath $installer -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=$prefix" -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "Miniforge setup failed (exit $($process.ExitCode)). See the installer output." }
    $condaExe = "$prefix\Scripts\conda.exe"
}
$condaBase = (& $condaExe info --base).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Conda could not be started. Repair your Conda installation and launch again.' }
$env:MPLCONFIGDIR = "$appDir\.runtime\cache\matplotlib"
& "$condaBase\python.exe" "$appDir\scripts\bootstrap.py" --conda "$condaExe" @args
exit $LASTEXITCODE
