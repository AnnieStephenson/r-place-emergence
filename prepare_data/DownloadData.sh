#!/bin/bash

get2022=false
get2023=true

# 2022
if $get2022
then
    #Atlas data
    wget https://raw.githubusercontent.com/placeAtlas/atlas/master/web/atlas.json

    #Description of source and data : https://www.reddit.com/r/place/comments/txvk2d/rplace_datasets_april_fools_2022/
    for filenum in {00..78}
    do
        wget https://placedata.reddit.com/data/canvas-history/2022_place_canvas_history-0000000000${filenum}.csv.gzip
        mv 2022_place_canvas_history-0000000000${filenum}.csv.gzip 2022_place_canvas_history-0000000000${filenum}.csv.gz
        gzip -d 2022_place_canvas_history-0000000000${filenum}.csv.gz
    done
fi

# 2023
if $get2023
then
    #Atlas data
    wget https://raw.githubusercontent.com/placeAtlas/atlas-2023/master/web/atlas.json
    #Description of source and data : https://www.reddit.com/r/place/comments/15bjm5o/rplace_2023_data/
    for filenum in {00..52}
    do
        wget https://placedata.reddit.com/data/canvas-history/2023/2023_place_canvas_history-0000000000${filenum}.csv.gzip
        mv 2023_place_canvas_history-0000000000${filenum}.csv.gzip 2023_place_canvas_history-0000000000${filenum}.csv.gz
        gzip -d 2023_place_canvas_history-0000000000${filenum}.csv.gz
    done
fi
