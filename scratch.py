import numpy as np

def read_lower_triangular_matrix(filename):
    """
    Reads a 20x20 lower triangular matrix from a file where each row
    has increasing number of space-separated values.
    Returns a full symmetric 20x20 numpy array.
    """
    
    matrix = np.zeros((20, 20))
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0  # Row index
    for line in lines:
        if i >= 20:
            break  # Only 20 rows needed
        values = list(map(float, line.strip().split()))
        for j, val in enumerate(values):
            matrix[i][j] = val
        i += 1

    # Mirror lower triangle to upper triangle for symmetry
    for i in range(20):
        for j in range(i+1, 20):
            matrix[i][j] = matrix[j][i]

    return matrix

def convert_to_flat_upper_triangle(matrix):
    """
    Converts a full symmetric 20x20 matrix into a JTT-style
    flat array of upper triangle values (i < j) in row-major order.
    """
    flat_rates = []
    for i in range(20):
        for j in range(i+1, 20):
            flat_rates.append(matrix[i][j])
    return flat_rates

# Example usage
filename = "wag.dat"  # Replace with your file path
matrix = read_lower_triangular_matrix(filename)
flat_vector = convert_to_flat_upper_triangle(matrix)

# Output result
print("JTT-style flat vector of 190 relative rates:")
for i, val in enumerate(flat_vector):
    print(f"{val:.6f}", end=", " if (i+1)%5 else ",\n")  # Print in readable chunks
