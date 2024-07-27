import numpy as np
from scipy.optimize import curve_fit

# Define the function to fit (example: exponential function)
def exp_func(x, a, b):
  return a * np.exp(b * x)

# Generate some sample data
x = np.linspace(0, 5, 100)  
y = exp_func(x, 2, -1)  
y += np.random.randn(len(y)) * 0.2

# Popt holds the optimized parameters, pcov is the covariance matrix
popt, pcov = curve_fit(exp_func, x, y)

# Print the optimized parameters
print("Optimized parameters:", popt)


fitted_curve = exp_func(x, *popt)  

# Plot the data and the fitted curve
import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Data')
plt.plot(x, fitted_curve, label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()