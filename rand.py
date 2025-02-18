import numpy as np

# Example 2D matrix
matrix = np.array([[1, 5, 3],
                   [7, 2, 9],
                   [4, 6, 8]])

# Number of top elements to find
k = 3

# Flatten the matrix
flattened = matrix.flatten()

# Get the indices of the top-k elements in the flattened array
# Using np.argpartition for better performance with large arrays
flattened_indices = np.argpartition(flattened, -k)[-k:]

# Sort the indices to get the top-k elements in order
flattened_indices_sorted = flattened_indices[np.argsort(flattened[flattened_indices])]

# Get the top-k elements
top_k_elements = flattened[flattened_indices_sorted]

# Convert the flattened indices back to 2D indices
row_indices, col_indices = np.unravel_index(flattened_indices_sorted, matrix.shape)

# Combine the row and column indices into a list of tuples
top_k_indices = list(zip(row_indices, col_indices))

print("Top-k elements:", top_k_elements)
print("Top-k indices (row, col):", top_k_indices)