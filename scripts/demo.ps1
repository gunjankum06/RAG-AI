param(
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$Collection = "demo",
    [string]$ApiKey = $env:API_KEY
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    Write-Error "API key missing. Set API_KEY env var or pass -ApiKey."
    exit 1
}

New-Item -ItemType Directory -Force -Path "data" | Out-Null
"RAG AI demo document. This project supports ingestion and question answering." | Out-File -FilePath "data/demo.txt" -Encoding utf8

Write-Host "Checking health..."
$health = Invoke-RestMethod -Method Get -Uri "$ApiUrl/health"
$health | ConvertTo-Json -Depth 5

Write-Host "Ingesting demo document..."
$ingestArgs = @(
    "--disable",
    "-sS", "-X", "POST", "$ApiUrl/api/v1/ingest",
    "-H", "X-API-Key: $ApiKey",
    "-F", "collection=$Collection",
    "-F", "files=@data/demo.txt"
)
$ingestRaw = & curl.exe @ingestArgs
$ingest = $ingestRaw | ConvertFrom-Json
$ingest | ConvertTo-Json -Depth 10

if ($ingest.status -ne "completed") {
    if ($ingest.message -match "/api/embed" -or $ingest.message -match "404") {
        Write-Host "Hint: check Ollama version/model compatibility and verify embedding endpoint support."
    }
    Write-Host "Ingest failed: $($ingest.message)"
    exit 1
}

$queryBody = @{
    question = "What is this demo document about?"
    collection = $Collection
    top_k = 5
    rerank = $true
    stream = $false
    chat_history = @()
} | ConvertTo-Json -Depth 5 -Compress

Write-Host "Querying demo collection..."
$query = Invoke-RestMethod -Method Post -Uri "$ApiUrl/api/v1/query" -Headers @{
    "X-API-Key" = $ApiKey
    "Content-Type" = "application/json"
} -Body $queryBody

$query | ConvertTo-Json -Depth 10

Write-Host "Demo complete."
