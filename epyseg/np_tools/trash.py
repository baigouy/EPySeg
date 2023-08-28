import numpy as np

# Create a 3D array with 100 rows
arr = np.random.rand(100, 3)

# Create a boolean index of length 100 that has only 5 True values
bool_index = np.zeros(100, dtype=bool)
bool_index[:5] = True
np.random.shuffle(bool_index)

print(bool_index.shape)
# Filter the rows of the 3D array using the boolean index
filtered_arr = arr[bool_index]

print(filtered_arr.shape)