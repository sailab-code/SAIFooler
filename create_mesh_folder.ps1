param($meshName)

mkdir -Path "./meshes/$meshName"

$jsonObj = @{
    "${meshName}_01" = @{
        "albedo" = "${meshName}_01_AlbedoTransparency.png";
        "normal" = "${meshName}_01_Normal.png";
        "metallic" = "${meshName}_01_MetallicSmoothness.png";
    }
}

$jsonObj | ConvertTo-Json | Out-File "./meshes/$meshName/${meshName}_mat_def.json"