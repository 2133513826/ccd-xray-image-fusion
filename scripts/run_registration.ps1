[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot '..\config\registration.yaml'),
    [string]$RunName = ''
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$executable = Join-Path $projectRoot 'build-new\register_objects.exe'
$ucrtBin = Join-Path $env:USERPROFILE 'scoop\apps\msys2\current\ucrt64\bin'
if (Test-Path -LiteralPath $ucrtBin -PathType Container) {
    $env:PATH = $ucrtBin + ';' + $env:PATH
}
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Registration executable not found: $executable. Build the project first."
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Registration config not found: $Config"
}

$registrationArguments = @((Resolve-Path -LiteralPath $Config).Path)
if ($RunName) {
    $registrationArguments += $RunName
}

& $executable @registrationArguments
if ($LASTEXITCODE -ne 0) {
    throw "Registration failed with exit code $LASTEXITCODE"
}
