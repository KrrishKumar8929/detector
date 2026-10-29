$ErrorActionPreference = 'Stop'
Write-Host 'Installing optional local background removal for the Pokémon Detector.' -ForegroundColor Cyan
Write-Host 'This uses the Python interpreter called by the service start script.' -ForegroundColor Yellow
python -m pip install 'rembg[cpu]'
if ($LASTEXITCODE -ne 0) {
    throw 'Installation failed. Check that this is the same Python environment used to run recognition_service.py.'
}
Write-Host 'Installation complete. Restart the Pokémon Detector service to enable segmentation.' -ForegroundColor Green
