param(
  [string]$HostUrl = "http://127.0.0.1:8000",
  [int]$Port = 8000,
  [switch]$NoStart
)

$ErrorActionPreference = "Stop"

function Get-EnvValue([string]$Key, [string]$Default="") {
  $envPath = Join-Path -Path (Get-Location) -ChildPath ".env"
  if (Test-Path $envPath) {
    $line = (Get-Content $envPath -ErrorAction SilentlyContinue | Where-Object { $_ -match "^$Key=" } | Select-Object -First 1)
    if ($line) {
      $val = $line -replace "^$Key=",""
      return $val.Trim()
    }
  }
  return $Default
}

function Invoke-Json([string]$Url, [string]$Method="GET", [hashtable]$Headers=@{}, [string]$Body="") {
  try {
    if ($Method -ieq "GET") { return Invoke-RestMethod -Method Get -Uri $Url -Headers $Headers -TimeoutSec 20 }
    elseif ($Method -ieq "POST") {
      if ($Body) { return Invoke-RestMethod -Method Post -Uri $Url -Headers $Headers -Body $Body -ContentType "application/json" -TimeoutSec 30 }
      else { return Invoke-RestMethod -Method Post -Uri $Url -Headers $Headers -TimeoutSec 30 }
    }
  } catch { return @{ error = $_.Exception.Message } }
}

$apiToken = Get-EnvValue -Key "API_TOKEN" -Default ""
if (-not $apiToken) {
  Write-Warning "API_TOKEN not found in .env. Protected endpoints may fail."
}

$uvicorn = ".\.venv\Scripts\uvicorn.exe"
if (-not (Test-Path $uvicorn)) {
  Write-Warning "uvicorn not found at $uvicorn. Ensure virtualenv is created and requirements installed."
}

if (-not $NoStart) {
  Write-Host "Stopping existing uvicorn (if any)..."
  Get-Process uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 1

  Write-Host "Starting server on $HostUrl ..."
  $args = "web_app.main:app --host 127.0.0.1 --port $Port"
  $proc = Start-Process -FilePath $uvicorn -ArgumentList $args -PassThru -WindowStyle Minimized
  Write-Host "uvicorn PID:" $proc.Id

  # wait for /health
  $ok=$false
  for ($i=0; $i -lt 20; $i++) {
    try {
      $resp = Invoke-RestMethod -Method Get -Uri "$HostUrl/health" -TimeoutSec 5
      if ($resp -eq "OK") { $ok=$true; break }
    } catch {}
    Start-Sleep -Milliseconds 500
  }
  if (-not $ok) { Write-Warning "Server health not confirmed. Continuing anyway..." }
}

# Prepare headers
$hdr = @{}
if ($apiToken) { $hdr["X-Api-Token"] = $apiToken }

# Log directory
$logDir = Join-Path (Get-Location) "tools\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "Calling diagnostics and builders..."

# Deep crawl (best-effort)
$deep = Invoke-Json -Url "$HostUrl/api/deep_crawl" -Method "POST" -Headers $hdr
$deep | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\deep_crawl.json"

# Build picks
$build = Invoke-Json -Url "$HostUrl/api/build_picks?token=$apiToken&limit=10" -Method "POST"
$build | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\build_picks.json"

# Top7 build
$top7b = Invoke-Json -Url "$HostUrl/api/top7_build?token=$apiToken&limit=7" -Method "POST"
$top7b | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\top7_build.json"

# Odds sources
$od = Invoke-Json -Url "$HostUrl/api/odds_sources?token=$apiToken"
$od | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\odds_sources.json"

# Match diagnostics
$diag = Invoke-Json -Url "$HostUrl/api/match_diagnostics?token=$apiToken"
$diag | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\match_diagnostics.json"

# Unmatched today
$umt = Invoke-Json -Url "$HostUrl/api/unmatched_today?token=$apiToken"
$umt | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\unmatched_today.json"

# Metrics
$met = Invoke-Json -Url "$HostUrl/api/metrics?token=$apiToken"
$met | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\metrics.json"

# Daily top5 and Top7
$top5 = Invoke-Json -Url "$HostUrl/api/daily_top5?explain=true"
$top5 | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\daily_top5.json"
$top7 = Invoke-Json -Url "$HostUrl/api/top7?token=$apiToken&hours_ahead=6"
$top7 | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 "$logDir\top7.json"

Write-Host "Done. Logs saved to $logDir"
