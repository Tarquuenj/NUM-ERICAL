def f(x):  # Define the function to integrate (replace with your own function)
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
  h = (b - a) / n  
  integral = 0.5 * (f(a) + f(b))  # Add contributions from endpoints

  for i in range(1, n):
    x = a + i * h
    integral += f(x)

  return integral * h  # Multiply by width to get total area

a = 0
b = 1
n = 1000  # Number of sub-intervals (more sub-intervals lead to higher accuracy)

result = trapezoidal_rule(f, a, b, n)

print("The approximate integral of f(x) from", a, "to", b, "using", n, "subintervals is:", result)