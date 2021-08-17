#Random Forest
#capturando os dados de um arquivo externo
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')

#atribuir nomes às colunas
df.columns =['f1', 'f2', 'f3', 'f4', 'label']

#label
labels = df.loc[:,['label']]
labels.head()

#separar cada uma das 4 features
features = []
for i in range(len(labels)):
  features.append([df['f1'][i], df['f2'][i], df['f3'][i], df['f4'][i]])

#Pareto 80/20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

#Treinando o modelo
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train.values.ravel())

#Predição dos dados usando X_test e guardando os dados para comparar com y_test, para saber se a acurácia está ok
y_pred=rfc.predict(X_test)

#Teste de acurácia, recall e matriz de confusão
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
print('Acurácia: ', accuracy_score(y_test,y_pred))
print('Recall: ', recall_score(y_test,y_pred))
print('Matriz de Confusão: \n', confusion_matrix(y_test,y_pred))

#Tentativa de Predizer um valor não inputado como treino nem teste
valor_predito = rfc.predict(X=[[-1.1111,1.1000,-1.2852,-1.3265]])
print('\nValor predito: ', valor_predito)
boolean_result = '';
if (valor_predito[0] == 1) :
  boolean_result = 'A nota é Verdadeira'
else :
  boolean_result = 'A nota é Falsa'

print('Conclusão: ', boolean_result)
