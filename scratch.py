import numpy as np

# Sample dataset
data = np.array([1, 2, 3, 4, 5])

# Calculate Interquartile Range (IQR) using numpy
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

print(q1)
print(q3)

print(iqr)