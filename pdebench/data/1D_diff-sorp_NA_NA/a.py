import h5py


# Open the HDF5 file
f = h5py.File("/home/shusrith/Downloads/a.h5", "r")

# Get all keys in the HDF5 file
keys = list(f.keys())

d = f["tensor"][0][0]
print(d.shape)