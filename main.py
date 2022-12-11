# Лінійна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

# print("Вводимо дані")
print(f"Подамо x як двовимірний масив:")
x = np.array([17.3,18.6,20.3,22.8,23.8,25.8,27.5,28.7,29.1,36.2]).reshape((-1, 1))
y = np.array([4.2,4.8,5.2,7.2,8.9,9.2,10.1,11.9,11.9,14.8])
# print("Створення самої моделі:")
model = LinearRegression()
print(model.fit(x, y))

# print("Отримаємо результати:")
# R² = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")

print(f"Подамо і y як двовимірний масив:")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept (a, 𝑏₀): {new_model.intercept_}")
print(f"slope (b, 𝑏₁): {new_model.coef_}")

# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) по формулі
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")
# модель для нових даних
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.white + f"Висновки:"
                 f"\n (𝑅²): {r_sq},"
                 f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏₁): {model.coef_}")

# Множинна регресія
# print("Вводимо дані")
y = [4.50, 5.15, 6.00, 5.55, 5.70, 6.55, 5.90, 6.15, 6.95, 6.40, 6.90, 7.35, 7.80, 8.00, 8.80, 9.30]

x = [
    [41.8, 7.2, 35.5], [44.6, 9.2, 34.4], [42.0, 11.6, 30.5], [49.3, 13.4, 40.1], [48.6, 13.4, 32.1],
    [54.2, 14.8, 39.2], [62.2, 15.40, 35.1], [57.3, 15.8, 39.3], [54.1, 16.2, 39.2], [69.4, 16.5, 41.1],
    [60.2, 17.0, 42.5], [65.2, 17.1, 45.2], [70.1, 18.0, 45.8], [75.5, 18.5, 43.9], [74.9, 19.0, 50.5],
    [72.3, 20.5, 48.3],
]
x, y = np.array(x), np.array(y)


# print("Створення самої моделі:")
model = LinearRegression().fit(x, y)
print(model)


# print("Отримаємо результати:")
# R² = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")


# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) по формулі
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(10).reshape((-1, 2))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.white + f"Висновки:"
                 f"\n (𝑅²): {r_sq},"
                 f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏₁): {model.coef_}")

