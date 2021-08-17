#Árvore de Decisão

#Dados
temperature = [5,    8,     10,   12,   14,    14,    18,    21,    21,     10,    23,   24,   24,    25,    25,   26,     26,    26,    27,    27,   28,    28,    30,    32,    32] #celsius
rainfall    = [0,    25,    0,    2,    0,     4,     0,     0,     7,      20,    5,    0,    12,    35,     7,    0,     23,    25,    32,    0,    3,     4,     0,     2,     5] #mm/hora
wind_speed  = [5,    8,     2,    22,   12,    8,     0,     58,    62,     25,    18,   5,    12,    35,     7,    0,     23,    25,    32,    0,    3,     4,     0,     5,     2] #km/h
did_bike    = [True, False, True, True, True,  True,  True,  False, False,  False, True, True, True,  False,  True, True,  False, False, False, True, False, False, False, False, False]

weather_conditions = []
for i in range(len(temperature)): 
  weather_conditions.append([temperature[i], rainfall[i], wind_speed[i]]) 

X = weather_conditions # Features
y = did_bike #Labels

#Decision Treee Classifier - treinando
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

#Passando um valor novo para ver a decisão
weather_condition = [[8, 25, 8]]
should_bike = clf.predict(weather_condition)
print('should_bike', should_bike)
