$threshes = (0.0, 0.05, 0.2)
$batches = ("1st", "2nd", "3rd", "4th")

foreach($batch in $batches) {
    $batch = "${batch}_batch.json"
    foreach($thresh in $threshes) {
        Write-Output "Launching $batch with threshold $thresh"
        python generate_saliency_plots.py --meshes_definition $batch --classifier inception --saliency-threshold $thresh --texture-rescale 0.33
    }
}