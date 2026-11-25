# Lab helper for Windows Terminal / PowerShell. Needs git on PATH (Git for Windows, or MSYS2: pacman -S git).
# Run from the lab root folder:
#   powershell -ExecutionPolicy Bypass -File scripts\bug.ps1 apply 7
#   powershell -ExecutionPolicy Bypass -File scripts\bug.ps1 reset 7
#   powershell -ExecutionPolicy Bypass -File scripts\bug.ps1 check
param([string]$cmd, [int]$n = 0)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\code")
$id = "{0:D2}" -f $n
switch ($cmd) {
  "apply" { $p = Get-ChildItem "..\bugs\B${id}_*.patch" | Select-Object -First 1
            git apply --verbose $p.FullName; if ($LASTEXITCODE -eq 0) { Write-Host "bug $id applied: cmake --build build -j 8 ; .\build\lab_runner.exe $n" } }
  "reset" { $p = Get-ChildItem "..\bugs\B${id}_*.patch" | Select-Object -First 1
            git apply -R --verbose $p.FullName; if ($LASTEXITCODE -eq 0) { Write-Host "bug $id removed" } }
  "check" { foreach ($p in Get-ChildItem "..\bugs\B*.patch") { git apply -R --check $p.FullName 2>$null; if ($LASTEXITCODE -eq 0) { Write-Host "APPLIED: $($p.Name)" } }
            Write-Host "(nothing listed = clean code)" }
  default { Write-Host "usage: scripts\bug.ps1 apply|reset N   or   scripts\bug.ps1 check" }
}
