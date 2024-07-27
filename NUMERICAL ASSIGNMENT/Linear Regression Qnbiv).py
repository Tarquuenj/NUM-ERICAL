import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5  # 100 random values
y = 2 * X + np.random.randn(100) * 0.5 + 5
data = pd.DataFrame({'Hours of Study': X, 'Score': y})
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Hours of Study')
plt.ylabel('Score')
plt.title('Linear Regression: Hours of Study vs Score')
plt.legend()
plt.show()
