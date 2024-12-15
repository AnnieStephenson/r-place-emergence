# r-place-emergence
Python package for exploring emergent collective phenomena in the 2022 r/place dataset

# Condensed Dataset info ('PixelChangesCondensedData_sorted.npz')
- reading vertically all horizontal arrays (at a common index) gives info about a given pixel change.
- 'eventNumber' array: eventNb//1e7 gives the number of the original file it comes from. The last 6 figures gives the order in which it appeared in that file (a file never contains more than 1e7 events)
- 'userIndex' points to a unique index for each user. 'userIDsFromIdx.json' gives the translation from index to full (string) user ID
- Similarly, 'colorIndex' points to the index of one of the 32 used colors. 'ColorsFromIdx.json' translates this index into hex colors
- 'seconds' are floats containing both the seconds and the milliseconds (three decimals places). Quantities <1e-3 in these numbers are nonsensical.
- 'pixelXpos' and 'pixelYpos' contain the X and Y positions of the pixel change
- 'moderatorEvent' contains a boolean saying if this pixel change comes from a moderator intervention (changing a whole rectangle into white)
- The arrays are sorted in terms of the 'seconds', starting at second 0.315.


# Requirements
In addition to installing in a standard way (pip install ```module```) the modules listed in requirements.txt, do:
```git clone git@github.com:martiniani-lab/sweetsourcod.git
python setup.py build_ext -i
```
and then add sweetsourcod_builddirectory/sweetsourcod/ to your PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:~/r-place-emergence/sweetsourcod/:~/r-place-emergence/sweetsourcod/sweetsourcod/
```

# For EWS paper: Analysis steps and corresponding scripts and output files
1. Dowload all original data with  
&rarr; Run `source prepare_data/DownloadData.sh`

2. Set the year of the r/place event to be studied in `__init__.py`, in the line `var = GlobalVars(year=2022)`. Both 2022 and 2023 years must be ran, for all following steps.

3. **Condense** the original data in a more compact form, i.e. pixel changes as rows and important info about each change as columns. Possibility to run for a few files of the original dataset at a time, before merging them. Then, the pixel changes are sorted by increasing time, duplicate pixel changes are removed, and some checks are ran. The output is a structured array in file `PixelChangesCondensedData_sorted.npz`, whose structure is described above.  
&rarr; Run `python prepare_data/run_condenser.py`.  
This mostly calls `prepare_data/CondenseData.py`.  
This also creates dictionaries of color indices to hex codes (`ColorsFromIdx.json`), and of user indices to user hashtags (`userIDsFromIdx.json`) and vice-versa (`userDict.json`).

4. Make very few corrections to **inconsistencies** in the atlas text file.
&rarr; Run `python prepare_data/correct_atlas.py`

5. Create all canvas compositions: one CanvasPart object per entry in the atlas, containing pixel changes info for all the time span and extent of this composition listed in the atlas. The needed input is the pixel changes file and the atlas. This also separates in distinct compositions the atlas entries that are disjoint in time or in space on the canvas.  
&rarr; Run `python prepare_data/run_all_cparts.py`.  
This calls mostly `CanvasPart()` from `rplacem/canvas_part.py`. The output is a pickle file of the form `data/[year]/canvas_comps_[year].pkl`

6. Clean the compositions (mostly: remove the compositions with same borders but two entries in the atlas + change name of compositions that are different but have the same atlas id).  
&rarr; Run `python prepare_data/clean_comps.py`.  
The output is of the form `data/[year]/canvas_comps_[year]_clean.pkl`

7. Build one CanvasPartStatistics object per composition, mostly containing time series for many variables describing the compositions. The input is the file with all CanvasPart objects. These objects should be generated with all parameters of the sensitivity study.   
&rarr; Run `python prepare_data/cpart_stat_param.py`.   
It is advised to run this in grid jobs, and then to merge output files with `python prepare_data/combine_cpart_stats.py`.   
The pickle output is of the form `cpart_stats_[param].pkl` where `[param]` is a string listing the global parameters of this specific sensitivity run.  

8. For each composition and each sensitivity run, build the training dataset for the XGBoost algorithm. A preliminary run  
`python rplacem/machine_learning/build_dataset.py -r True`  
is needed to compile all time instances that will be rejected because they are duplicates of a time instance of another composition with high spatial overlap. That preliminary run produces a file `reject_times_from_overlapping_comps_[param].pickle`, which is then used in the main run:  
&rarr; Run `python rplacem/machine_learning/build_dataset.py`  
The option `-p [num]` (e.g. `-p 0` for the nominal parameters) of this script can be used to run a specific sensitivity run of number \[num\].

9. Run the training and evaluation with an XGBoost algorithm, for each sensitivity analysis:  
&rarr; Run `python rplacem/machine_learning/xgboost_regression.py -p [num]`  
This should be ran first for the default parameter `--test2023 False` to train and evaluate on the 2022 dataset, then with `--test2023 True` to train on 2022 and evaluate on 2023 data. The option `--shapplots True` can be used to generate multiple SHAPley plots.



# Testing
In the terminal, navigate to the rplacem directory and run:
```shell
pytest
```