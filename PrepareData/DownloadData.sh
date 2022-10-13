#!/bin/bash

#Atlas data
wget https://raw.githubusercontent.com/placeAtlas/atlas/master/web/atlas.json

#Description of source and data : https://www.reddit.com/r/place/comments/txvk2d/rplace_datasets_april_fools_2022/
for filenum in {00..78}
do
    wget https://placedata.reddit.com/data/canvas-history/2022_place_canvas_history-0000000000${filenum}.csv.gzip
    mv 2022_place_canvas_history-0000000000${filenum}.csv.gzip 2022_place_canvas_history-0000000000${filenum}.csv.gz
    gzip -d 2022_place_canvas_history-0000000000${filenum}.csv.gz
done
