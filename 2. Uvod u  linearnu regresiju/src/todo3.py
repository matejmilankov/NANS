# TODO 3: implementirati primenu jednostavne linearne regresije
# nad podacima iz datoteke "data/skincancer.csv".
import pandas as pd
import linreg_simple as ls
import matplotlib.pyplot as plt
df = pd.read_csv('../data/skincancer.csv')

x = df['Lat']
y = df['Mort']

slope, intercept = ls.fit(x, y)
ls.predict(x, slope, intercept)
y_pred = ls.make_predictions(x, slope, intercept)

plt.plot(x, y, 'xr')
plt.plot(x, y_pred, 'b')
plt.title(f'slope: {slope:.2f}, intercept: {intercept:.2f}')
plt.show()

