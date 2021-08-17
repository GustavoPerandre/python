import pandas as pd

#captura dos dados que serão usado
money_data_frame = pd.read_csv("https://drive.google.com/uc?id=1FtG5iX48UGSi7V8CTf_luw7T8GCsu8kI")
happiness_data_frame = pd.read_csv("https://drive.google.com/uc?id=1A4PzcQLIM6i2JSmp3WPAbEDUaxMAfMJf")

#Merge dos dataframes
money_happiness_df = pd.merge(money_data_frame, happiness_data_frame, on='Country', how='inner')

#ordenação
#money_happiness_df = money_happiness_df.sort_values(by=['GDP'], ascending=False)

#conversão para float para que possa ser usado pelo modelo de regressão
money_happiness_df['GDP'] = money_happiness_df['GDP'].apply(float)

#Conversão da array em uma matriz (pois a feature precisa ser uma matriz)
import numpy as np
X = np.array(money_happiness_df['GDP']).reshape(-1,1) #matriz
y = money_happiness_df['Happiness'] #array

#IA
import sklearn.linear_model
linear_regression_model = sklearn.linear_model.LinearRegression()
linear_regression_model.fit(X, y)
sklearn.linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#Apresentação dos dados
import matplotlib.pyplot as plt
prediction_y = linear_regression_model.predict(X)
plt.scatter(X, y,  color='blue')
plt.plot(X,prediction_y, color='red')
plt.show()

#Predizendo a satisfação com a vida de um país onde eu só tenho o Produto Interno Bruto (GPD), usando o modelo que criamos
argentina_gpd = [[13588.84]] #array de duas dimensões
linear_regression_model.predict(argentina_gpd)
