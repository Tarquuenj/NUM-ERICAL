import numpy as np
from scipy.interpolate import CubicSpline

# Sample data (replace with your own data)
x = np.array([0, 1, 2, 3])
y = np.array([1, 2, 1, 3])

# Create a cubic spline object
spline = CubicSpline(x, y)

# Evaluate the spline at new points
new_x = np.linspace(0, 3, 100)  # Create dense points for smooth curve
new_y = spline(new_x)

# Plot the data points and the interpolated curve
import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Data Points')
plt.plot(new_x, new_y, label='Spline Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Get the interpolated value at a specific point
specific_x = 1.5
interpolated_y = spline(specific_x)

print("Interpolated value at x =", specific_x, ":", interpolated_y)