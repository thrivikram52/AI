import numpy as np

# 1. Creating Arrays
# Creating a 1D array
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:\n", array_1d)

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", array_2d)

# 2. Array Attributes
print("\nArray Shape:", array_2d.shape)
print("Array Data Type:", array_2d.dtype)
print("Number of Dimensions:", array_2d.ndim)
print("Array Size (number of elements):", array_2d.size)

# 3. Creating Special Arrays
# Array of zeros
zeros_array = np.zeros((3, 4))
print("\nArray of Zeros:\n", zeros_array)

# Array of ones
ones_array = np.ones((2, 3))
print("\nArray of Ones:\n", ones_array)

# Identity matrix
identity_matrix = np.eye(3)
print("\nIdentity Matrix:\n", identity_matrix)

# Array with a range of values
range_array = np.arange(0, 10, 2)
print("\nRange Array (0 to 10 with step 2):\n", range_array)

# Array with evenly spaced values
linspace_array = np.linspace(0, 1, 5)
print("\nLinspace Array (0 to 1 with 5 values):\n", linspace_array)

# 4. Reshaping Arrays
reshaped_array = np.arange(1, 13).reshape((3, 4))
print("\nReshaped Array (3x4):\n", reshaped_array)

# 5. Array Indexing and Slicing
# Indexing
print("\nElement at index 1 in 1D array:", array_1d[1])
print("Element at row 1, column 2 in 2D array:", array_2d[1, 2])

# Slicing
print("\nSlicing first three elements in 1D array:", array_1d[:3])
print("Slicing first two rows and columns 1 to 2 in 2D array:\n", array_2d[:2, 1:3])

# 6. Mathematical Operations
# Element-wise operations
print("\nArray Addition:\n", array_1d + 2)
print("Array Multiplication:\n", array_1d * 2)

# Array-wise operations
print("\nArray Sum:", array_1d.sum())
print("Array Mean:", array_1d.mean())

# Matrix multiplication
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
matrix_product = np.dot(matrix1, matrix2)
print("\nMatrix Product:\n", matrix_product)

# 7. Statistical Operations
random_array = np.random.rand(5)
print("\nRandom Array:\n", random_array)
print("Max Value:", random_array.max())
print("Min Value:", random_array.min())
print("Standard Deviation:", random_array.std())

# 8. Stacking and Splitting Arrays
# Stacking arrays vertically and horizontally
array_a = np.array([[1, 2], [3, 4]])
array_b = np.array([[5, 6], [7, 8]])

vertical_stack = np.vstack((array_a, array_b))
print("\nVertical Stack:\n", vertical_stack)

horizontal_stack = np.hstack((array_a, array_b))
print("\nHorizontal Stack:\n", horizontal_stack)

# Splitting arrays
split_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
split1, split2 = np.hsplit(split_array, 2)
print("\nHorizontal Split:\n", split1, "\n", split2)

# 9. Boolean Masking and Advanced Indexing
# Boolean masking
bool_mask = array_1d > 3
print("\nBoolean Mask:", bool_mask)
print("Filtered Array (elements > 3):", array_1d[bool_mask])

# Advanced indexing
advanced_indexing = array_2d[[0, 1], [1, 2]]  # Select elements (0,1) and (1,2)
print("\nAdvanced Indexing (elements at [0,1] and [1,2]):", advanced_indexing)

# 10. Saving and Loading Arrays
# Saving an array to a file
np.save('saved_array.npy', array_1d)
print("\nArray saved to 'saved_array.npy'.")

# Loading an array from a file
loaded_array = np.load('saved_array.npy')
print("Loaded Array:\n", loaded_array)
