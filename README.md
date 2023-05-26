# r-place-emergence
Python package for exploring emergent collective phenomena in the 2022 r/place dataset

# Treated Dataset info ('PixelChangesCondensedData_sorted.npz')
- reading vertically all horizontal arrays (at a common index) gives info about a given pixel change.
- 'eventNumber' array: eventNb//1e7 gives the number of the original file it comes from. The last 6 figures gives the order in which it appeared in that file (a file never contains more than 1e7 events)
- 'userIndex' points to a unique index for each user. 'userIDsFromIdx.json' gives the translation from index to full (string) user ID
- Similarly, 'colorIndex' points to the index of one of the 32 used colors. 'ColorsFromIdx.json' translates this index into hex colors
- 'seconds' are floats containing both the seconds and the milliseconds (three decimals places). Quantities <1e-3 in these numbers are nonsensical.
- 'pixelXpos' and 'pixelYpos' contain the X and Y positions of the pixel change
- 'moderatorEvent' contains a boolean saying if this pixel change comes from a moderator intervention (changing a whole rectangle into white)
- The arrays are sorted in terms of the 'seconds', starting at second 0.315.

#### For testing
In the terminal, navigate to the rplacem directory and run:
```shell
pytest
```

#### Requirements
In addition to installing in a standard way (pip install ```module```) the modules listed in requirements.txt, do:
```git clone git@github.com:martiniani-lab/sweetsourcod.git
python setup.py build_ext -i
```
and then add sweetsourcod_builddirectory/sweetsourcod/ to your PYTHONPATH
