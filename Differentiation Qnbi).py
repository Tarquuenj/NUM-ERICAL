from sympy import Symbol, diff
# Define a symbolic variable
x = Symbol('x')
# Define a function
def f(x):
  return x**2 + 3*x - 4
# Differentiate the function
df = diff(f(x), x)
# Print the derivative
print(df)