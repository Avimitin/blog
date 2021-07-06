---
title: 'Useful Script For ffmpeg'
author: avimitin
date: 2021/06/30 12:57
tag: [ffmpeg]
---
# Some useful ffmpeg scripts

Scripts below are really useful in daily life.

## MP4 to GIF

In PowerShell you will need to replace the excape character.

```console
# Bash Only
ffmpeg -i {{ MP4 FILENAME }} \
      -filter_complex \
      "[0:v] fps=12,scale=w=480:h=-1,split [a][b];[a] \
      palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" \
      {{ GIF FILENAME }}

# PowerShell
ffmpeg -i {{ MP4 FILENAME }} `
      -filter_complex `
      "[0:v] fps=12,scale=w=480:h=-1,split [a][b];[a] `
      palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" `
      {{ GIF FILENAME }}
```

## Shrink Size

```console
ffmpeg -i "%~1" \
       -filter:v "scale=-1:'min(1080,ih)'" \
       -movflags +faststart \
       -crf 24 \
       -preset faster \
       -c:v libx265 \
       -pix_fmt yuv420p \
       -flags  +loop \
       -x265-params "bframes=10:ref=5" \
       -deblock 0:0 \
       -map "0:v:0" -map "0:a:0" \
       -c:a libfdk_aac \
       -vbr 5 \
       "%~dpn1-ENCODE.mp4"
```
