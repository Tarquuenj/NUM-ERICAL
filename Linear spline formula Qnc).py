# Given data points  
X = [2.00, 4.25, 5.25, 7.81, 9.20, 10.60]  
Y = [7.2, 7.1, 6.0, 5.0, 3.5, 5.0]  

# Points for interpolation  
x1 = 2.00  
y1 = 7.2  
x2 = 4.25  
y2 = 7.1  

# Value to interpolate  
x = 4.0  

# Linear interpolation formula  
y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)  

print(f"The value of y at x = {x} is approximately {y:.2f}")