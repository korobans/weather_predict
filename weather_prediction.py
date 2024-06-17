import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Заполнение пропусков методом интерполяции
df['Temperature'].interpolate(method='time', inplace=True)

# Агрегируем данные до средних значений за день
df_daily = df.resample('D').mean().dropna()

# Разделение данных на обучающую и тестовую выборки в соотношении 4:1
train_size = int(len(df_daily) * 0.8)
train, test = df_daily[:train_size], df_daily[train_size:]

# Настройки сезонности
season = 30

# Обучение модели SARIMA
model = SARIMAX(df_daily['Temperature'], order=(1, 0, 1), seasonal_order=(1, 1, 0, season))
model_fit = model.fit()

# Прогноз на будущее (следующие 7 дней)
forecast_future = model_fit.get_forecast(steps=7)

# Создаем новый DataFrame для будущих значений с корректным индексом
forecast_index = pd.date_range(start='2024-04-30', periods=7, freq='D')
forecast_df = pd.DataFrame({'Прогноз температуры': forecast_future.predicted_mean.values}, index=forecast_index)

print(forecast_df.head())

# Объединяем прогноз с исходными данными
merged_df = pd.concat([df, forecast_df])

# Визуализация исходных данных и прогноза
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index[:-7], merged_df['Temperature'][:-7], label='Исходные данные')
plt.plot(merged_df.index[-7:], merged_df['Прогноз температуры'][-7:], label='Прогноз')
plt.title('Прогноз средней дневной температуры')
plt.xlabel('Дата')
plt.ylabel('Температура (°C)')
plt.legend()
plt.grid(True)
plt.show()