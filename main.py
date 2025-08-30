from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt

x, y = fetch_california_housing(return_X_y=True)

model = KNeighborsRegressor()
model.fit(x, y)

pred=model.predict(x)

plt.scatter(pred , y)
plt.show()
