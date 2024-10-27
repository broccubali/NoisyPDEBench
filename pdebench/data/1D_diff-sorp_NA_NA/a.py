import h5py


# Open the HDF5 file
f = h5py.File("a.h5", "r")

# Get all keys in the HDF5 file
keys = list(f.keys())

print(f["0000"]["grid"]["x"][:])
