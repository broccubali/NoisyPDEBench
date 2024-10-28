import h5py


# Open the HDF5 file
f = h5py.File(
    "/home/shusrith/projects/blind-eyes/NoisyPDEBench/pdebench/data/1D_diff-sorp_NA_NA/a.h5",
    "r",
)

# Get all keys in the HDF5 file
keys = list(f.keys())

print(f["0000"][])
