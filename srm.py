import sqlite3
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Подключение к базе данных
conn = sqlite3.connect('voronezh.db')

# SQL-запрос для извлечения данных, исключая май и июнь 2024 года
query = "SELECT Date_Time, Temperature FROM weather_data WHERE (Date_Time NOT LIKE '__.06.2024%') AND (Date_Time NOT LIKE '__.05.2024%')"
df = pd.read_sql_query(query, conn)

# Закрытие соединения с базой данных
conn.close()

# Преобразуем 'Date_Time' в формат даты и установим его в качестве индекса
df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d.%m.%Y %H:%M')
df.set_index('Date_Time', inplace=True)

# Проверка на наличие NaN значений
print("Количество пропусков в данных:")
print(df.isna().sum())

df['Temperature'].interpolate(method='time', inplace=True)

# Проверка на наличие NaN значений после обработки
print("Количество пропусков в данных после обработки:")
print(df.isna().sum())

# Проверка различных значений периода сезонности
for m in [6, 12, 24]:
    print(f"Подбор модели с периодом сезонности: {m}")
    model = pm.auto_arima(df['Temperature'],
                          start_p=1, start_q=1,
                          max_p=3, max_q=3,
                          seasonal=True,
                          start_P=0, start_Q=0,
                          max_P=2, max_Q=2,
                          m=m,
                          d=None, D=1,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    print(model.summary())

# Сезонная декомпозиция временного ряда
stl = STL(df['Temperature'], seasonal=13)  # Попробуйте разные значения для параметра seasonal
result = stl.fit()

# Построение графиков сезонной декомпозиции
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
result.observed.plot(ax=ax1, title='Наблюдаемые данные')
result.trend.plot(ax=ax2, title='Тренд')
result.seasonal.plot(ax=ax3, title='Сезонность')
plt.tight_layout()
plt.show()