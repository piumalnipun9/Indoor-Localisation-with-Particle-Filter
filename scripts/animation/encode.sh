#!/bin/bash
# Encode the rendered frames to a LinkedIn-ready MP4 with three short fading captions.
set -e
cd "$(dirname "$0")/../../outputs/animation"
FONT=/usr/share/fonts/opentype/inter/Inter-SemiBold.otf
OUT=${1:-ble_tracking_30s.mp4}

caption() {  # text start end
    local a="if(lt(t,$2),0,if(lt(t,$2+0.5),(t-$2)/0.5,if(lt(t,$3-0.5),1,if(lt(t,$3),($3-t)/0.5,0))))"
    echo "drawtext=fontfile=$FONT:text='$1':fontsize=58:fontcolor=white:alpha='$a':x=96:y=h-150:shadowcolor=black@0.55:shadowx=0:shadowy=2"
}

FILTER="$(caption 'BLE beacons' 0.3 4.5),$(caption 'Particle filter' 6.0 11.5),$(caption 'Indoor positioning' 24.0 29.6),fade=t=in:st=0:d=0.6,fade=t=out:st=29.4:d=0.6"

ffmpeg -y -loglevel error -framerate 30 -i frames/%04d.png -vf "$FILTER" \
    -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart "$OUT"
echo "wrote $OUT"
