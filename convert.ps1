param ($fbxPath, $objPath, $texPath)


if($texPath -is [array])
{
    $texPath = $texPath -join " "
}

$command = "blender --background .\blender\workspace.blend --python .\blender\scripts\convert_fbx_to_obj.py -- $fbxPath $objPath $texPath"
Write-Host EXECUTING: $command -ForegroundColor White -BackgroundColor Cyan
Write-Host `n`n 

iex $command
