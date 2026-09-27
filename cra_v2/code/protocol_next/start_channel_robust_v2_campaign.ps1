param(
    [string]$Output = "",
    [int[]]$AllowedGpuPid = @(),
    [switch]$ScoreDevelopment
)
$ErrorActionPreference = "Stop"
$Project = "D:\3800"
$Distro = "VMUNet-Ubuntu22"
$Python = "/home/vmunet/venvs/widan/bin/python"

if ([string]::IsNullOrWhiteSpace($Output)) {
    # A frozen launcher lives at <campaign>/code/protocol_next.  Resolve its
    # campaign from that immutable location so a copied snapshot cannot point
    # back to an older campaign.  The live project launcher must be given an
    # explicit -Output because its parent layout is not a frozen campaign.
    $candidate = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
    if (-not (Test-Path -LiteralPath (Join-Path $candidate "plan.json"))) {
        throw "Non-frozen launcher requires an explicit -Output campaign path."
    }
    $Output = $candidate
}
$Output = [IO.Path]::GetFullPath($Output)
$Lock = Join-Path $Output "campaign.lock"

function Invoke-AuditedWsl([string]$Phase, [string[]]$Arguments) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $logPath = Join-Path $Output "logs\launcher-$stamp-$Phase.log"
    # Windows PowerShell promotes native stderr records to terminating errors
    # under the script-wide Stop policy.  Keep the child exit code authoritative
    # while still persisting the complete stdout/stderr transcript.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & wsl.exe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append | ForEach-Object { Write-Host $_ }
        $code = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    $event = @{
        created_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); phase=$Phase
        powershell_pid=$PID; exit_code=$code; transcript=$logPath; arguments=$Arguments
    } | ConvertTo-Json -Depth 5 -Compress
    [IO.File]::AppendAllText((Join-Path $Output "logs\launcher_events.jsonl"),
        $event + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    return $code
}

if (-not (Test-Path -LiteralPath $Output)) {
    throw "Prepare the frozen plan before starting fits: python protocol_next/run_channel_robust_v2_campaign.py prepare --output <path>"
}
$lockStream = $null
if (Test-Path -LiteralPath $Lock) {
    try { $stale = Get-Content -LiteralPath $Lock -Raw | ConvertFrom-Json }
    catch { throw "Campaign lock is unreadable; refusing automatic removal: $Lock" }
    [long]$stalePid = 0
    $pidValid = [long]::TryParse([string]$stale.powershell_pid, [ref]$stalePid)
    if ($stale.output -ne $Output -or -not $pidValid -or $stalePid -le 0) {
        throw "Campaign lock payload/output differs; refusing automatic removal: $Lock"
    }
    $owner = Get-Process -Id $stalePid -ErrorAction SilentlyContinue
    $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [long]$stale.created_unix
    if ($owner) { throw "Campaign lock owner PID $stalePid is still alive." }
    if ($age -lt 300) { throw "Campaign lock owner is absent but lock is younger than 300 seconds; retry later." }
    Remove-Item -LiteralPath $Lock -Force
    $event = @{ created_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); phase='stale_lock_recovered'
        stale_owner_pid=$stalePid; lock_path=$Lock; output=$Output } | ConvertTo-Json -Compress
    [IO.File]::AppendAllText((Join-Path $Output "logs\launcher_events.jsonl"),
        $event + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}
try {
    $lockStream = [System.IO.File]::Open($Lock, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    $lockPayload = [Text.Encoding]::UTF8.GetBytes((@{
        powershell_pid = $PID; created_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        output = $Output
    } | ConvertTo-Json -Compress))
    $lockStream.Write($lockPayload, 0, $lockPayload.Length); $lockStream.Flush()

    $OutputWsl = (($Output -replace '^D:', '/mnt/d').Replace('\', '/'))
    $Runner = "$OutputWsl/code/protocol_next/run_channel_robust_v2_campaign.py"
    while ($true) {
        $status = Get-Content -LiteralPath (Join-Path $Output "fit_state.json") -Raw | ConvertFrom-Json
        if ($status.status -eq "completed") { break }

        $os = Get-CimInstance Win32_OperatingSystem
        $ramMiB = [math]::Floor([double]$os.FreePhysicalMemory / 1024)
        $gpu = @(nvidia-smi --query-gpu=utilization.gpu,memory.free,memory.total --format=csv,noheader,nounits)
        $computeRaw = @(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)
        $compute = @()
        foreach ($line in $computeRaw) {
            if ($line -match '^\s*(\d+)\s*,\s*(.*?)\s*,\s*(\d+)\s*$') {
                $compute += @{ pid=[int]$Matches[1]; name=$Matches[2]; used_gpu_memory_mib=[int]$Matches[3] }
            }
        }
        $unknown = @($compute | Where-Object { $AllowedGpuPid -notcontains $_.pid })
        $gpuUtil = [int](($gpu[0] -split ',')[0].Trim())
        $gpuFree = [int](($gpu[0] -split ',')[1].Trim())
        $approved = ($ramMiB -ge 4096 -and $gpuFree -ge 2048 -and $gpuUtil -le 40 -and $unknown.Count -eq 0)
        $audit = @{
            created_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); powershell_pid=$PID
            lock_path=$Lock; windows_free_ram_mib=$ramMiB; gpu_rows=$gpu
            compute_processes=$compute; allowed_gpu_pids=$AllowedGpuPid; unknown_compute_processes=$unknown
            approved=$approved
        }
        $auditPath = Join-Path $Output "logs\preflight_current.json"
        [IO.File]::WriteAllText($auditPath, ($audit | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
        [IO.File]::AppendAllText((Join-Path $Output "logs\preflight_history.jsonl"),
            ($audit | ConvertTo-Json -Depth 6 -Compress) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        if (-not $approved) {
            Write-Host "Resource gate paused: RAM=$ramMiB MiB, GPU free=$gpuFree MiB, util=$gpuUtil%, unknown compute processes=$($unknown.Count)."
            exit 3
        }
        $auditWsl = (($auditPath -replace '^D:', '/mnt/d').Replace('\', '/'))
        $code = Invoke-AuditedWsl "fit-one" @('-d',$Distro,'--','env',"CRA_V2_LAUNCH_AUDIT=$auditWsl",$Python,$Runner,'fit-one','--output',$OutputWsl)
        if ($code -ne 0) { exit $code }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Output "predev_seal.json"))) {
        $code = Invoke-AuditedWsl "seal" @('-d',$Distro,'--',$Python,$Runner,'seal','--output',$OutputWsl)
        if ($code -ne 0) { exit $code }
    }
    if ($ScoreDevelopment) {
        while ($true) {
            $dev = Get-Content -LiteralPath (Join-Path $Output "dev_state.json") -Raw | ConvertFrom-Json
            if ($dev.status -eq "completed") { break }
            $code = Invoke-AuditedWsl "dev-one" @('-d',$Distro,'--',$Python,$Runner,'dev-one','--output',$OutputWsl)
            if ($code -ne 0) { exit $code }
        }
        $code = Invoke-AuditedWsl "report" @('-d',$Distro,'--',$Python,$Runner,'report','--output',$OutputWsl)
        exit $code
    }
    Write-Host "All fits are sealed. Development scoring remains stopped; rerun with -ScoreDevelopment when authorized."
}
finally {
    if ($lockStream) { $lockStream.Dispose() }
    if (Test-Path -LiteralPath $Lock) { Remove-Item -LiteralPath $Lock -Force }
}
