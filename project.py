import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split

np.warnings.filterwarnings('ignore')

df = pd.read_excel('branch_data.xlsx')

df2 = df[['BRANCH_ID', 'NO_COST_MAY', 'LOW_COST_MAY', 'HIGH_COST_MAY', 'NO_COST_APR', 'LOW_COST_APR', 'HIGH_COST_APR']]

# ********** Total account and Total Amount **************************

df_sum_ac = pd.DataFrame(data=[["APR", df.TOT_AC_APR.sum()],
                               ["MAY", df.TOT_AC_MAY.sum()]], columns=['Month', 'TOTAL'])

fig = px.bar(df_sum_ac, x='Month', y='TOTAL', text='TOTAL', color='Month')
fig.show()

df_sum_amt = pd.DataFrame(data=[["APR", df.TOT_AMT_APR.sum()],
                               ["MAY", df.TOT_AMT_MAY.sum()]], columns=['Month', 'TOTAL'])

fig = px.bar(df_sum_amt, x='Month', y='TOTAL', text='TOTAL', color='Month')
fig.show()

# ********** Subplots **************************

df_br = df['BRANCH_ID']
df_may = df['TOT_AMT_MAY']
df_apr = df['TOT_AMT_APR']
df_may_ac = df['TOT_AC_MAY']
df_apr_ac = df['TOT_AC_APR']

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Total Account Growth", "Total Amount Growth"))

# fig = go.Figure()


fig.add_trace(
    go.Bar(x=df_br,
           y=df_may_ac, name='May_AC'), row=1, col=1
           )

fig.add_trace(
    go.Bar(x=df_br,
           y=df_apr_ac, name='Apr_AC'), row=1, col=1
           )

fig.add_trace(
    go.Bar(x=df_br,
           y=df_may, name='May'), row=2, col=1
           )

fig.add_trace(
    go.Bar(x=df_br,
           y=df_apr, name='Apr'
           ), row=2, col=1)

fig.show()

#Prepared dataset overall plotting using Seaborn

plt.figure(figsize=(8,8))
sns.pairplot(df2, diag_kind='hist', hue='BRANCH_ID')
plt.savefig('br_data.png')
plt.clf()


# Use plotly to draw the Heatmap.

trace = go.Heatmap(z=df.corr().round(2),
                   x=df.columns,
                   y=df.columns)
plot([trace], filename='plotly-heatmap.html', )

# Applying ML Model

X = df.drop('TOT_AMT_MAY', axis=1)
y = df['TOT_AMT_MAY']

# X = titanic.drop('Survived', axis=1)
# y = df['TOT_AMT_MAY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lm.predict(X_test)

from sklearn.metrics import f1_score

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="inferno")

# Plotting differenct between real and predicted values
sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

# Plotting the residuals: the error between the real and predicted values
residuals = y_test - predicted_values
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plt.show()

# Understanding the error that we want to minimize
from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
print(f"R2 Score: {metrics.r2_score(y_test, predicted_values)}")

# print('Overall f1-score')
# print(f1_score(y_test, predicted_values, average="macro"))
#
# print('Coefficients')
# print(pd.DataFrame(lm.coef_, columns=X.columns).to_string())
