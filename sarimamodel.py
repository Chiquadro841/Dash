import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Caricare i dati dal file CSV
data = pd.read_csv('sp500_monthly_data.csv')

# Convertire la colonna 'Date' in formato datetime e impostarla come indice
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index('Date', inplace=True)

# Creare e addestrare il modello SARIMAX
# Usare un ordine (p,d,q) e stagionalità (P,D,Q,s) adatti. Qui usiamo valori arbitrari (1,1,1) per esempio.
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

model = SARIMAX(data['Close'], 
                exog=data[['Eps', 'FedFundsRate']], 
                order=order, 
                seasonal_order=seasonal_order)
results = model.fit()

# Creare un DataFrame per le previsioni future
future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
future_exog = pd.DataFrame({'Eps': [200]*12, 'FedFundsRate': [4.5]*12}, index=future_dates)

# Fare la previsione
forecast = results.get_forecast(steps=12, exog=future_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Visualizzare i risultati
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Storico')
plt.plot(future_dates, forecast_mean, label='Previsione')
plt.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Previsione Close con SARIMAX')
plt.legend()
plt.show()

# Stampare le previsioni
print(forecast_mean)
