def f(x):  # Define the function to integrate
  return x**2

def trapezoidal_rule(f, a, b, n):
  """
  This function computes the numerical integral of f(x) from a to b using the Trapezoidal Rule.

  Args:
      f: The function to integrate.
      a: The lower bound of integration.
      b: The upper bound of integration.
      n: The number of subintervals.

  Returns:
      The approximate value of the integral.
  """
  h = (b - a) / n  # Calculate the width of each subinterval
  integral = 0.5 * (f(a) + f(b))  # Add contributions from endpoints

  # Loop through interior points and add their contribution
  for i in range(1, n):
    x = a + i * h
    integral += f(x)

  return integral * h  # Multiply by width to get total area
a = 0
b = 1
n = 100  

result = trapezoidal_rule(f, a, b, n)

print("The approximate integral of f(x) from", a, "to", b, "using", n, "sub-intervals is:", result)