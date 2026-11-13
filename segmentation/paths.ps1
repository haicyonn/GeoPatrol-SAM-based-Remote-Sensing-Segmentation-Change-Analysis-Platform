function ConvertTo-WslArtifactPath {
    param([Parameter(Mandatory=$true)][string]$ArtifactPath)
    if($ArtifactPath -notmatch '^[A-Za-z]:[\\/]'){
        throw 'A fully qualified Windows drive path is required.'
    }
    return $ArtifactPath.Replace('\','/')
}
