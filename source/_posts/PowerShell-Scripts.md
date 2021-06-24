---
author: avimitin
date: 2021/06/24 21:32
tag: [windows, powershell]
categories: [windows]
---
# PowerShell Scripts

Here stores some of the useful script.

## Shrink video sizes

Find all the file with `*.MP4` extension and use ffmpeg to pressing it.

```powershell
Get-ChildItem -Path .\ -Include *.MP4 -Exclude *-shrink.MP4 -Recurse | ForEach-Object {
	.\ffmpeg.exe -i $_.Name -filter:v "scale=-1:'min(1080,ih)'" -movflags +faststart -crf 24 -preset faster -c:v libx265 -pix_fmt yuv420p -flags  +loop -x265-params "bframes=10:ref=5" -deblock 0:0 -map "0:v:0" -map "0:a:0" -c:a libfdk_aac -vbr 5 ($_.BaseName + "-shrink.mp4")
}
```

Replace `*.MP4` with `*.MKV` to pressing mkv file. Replace `*.MP4` with `*.MKV,*.MP4`
to pressing both extension.

## Find and replace batch

I've made some error when executed the script above: I named the compressed file 
with `-shirnk.MP4`. So I write a script to replace extension.

```powershell
Get-ChildItem -Path .\ -Include *shirnk.MP4 -Recurse | ForEach-Object {
	Rename-Item $_.Name ($_.BaseName.Split('-')[0] + '-shrink' + $_.Extension)
}
```

Replace `*shirnk.MP4` with the file name wild card you want.

