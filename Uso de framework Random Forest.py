#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importar librerias
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def intro():
    print("----------------------------------------------------------------------------------------\n")
    print("      Modelo para predecir si una persona tiene o no una enfermedad cardiovascular      \n")
    print("                   Realizado por: Nancy Lesly Segura Cuanalo A01734337                  \n")
    print("----------------------------------------------------------------------------------------\n")


# In[ ]:


def show_variables():
    print("Las variables presentes en este modelo son: \n")
    for i in range(11):
        print(i+1," -> ",X.columns[i])
    print("----------------------------------------------------------------------------------------\n")


# In[ ]:


def final():
    print("----------------------------------------------------------------------------------------\n")
    print("Fin de la ejecución")
    print("Gracias por utilizar este modelo predictivo")
    print("----------------------------------------------------------------------------------------\n")


# In[ ]:


#Cargar datos 
#Se utilizará el dataset de kaggle: https://www.kaggle.com/code/seowkhaiwen/cardiovascular-analysis/data
#Son datos sobre personas que pueden o no tener alguna enfermedad cardiovascular
data=pd.read_csv('https://raw.githubusercontent.com/Nancy2405/UsoBiblioteca_Clasificacion/main/cardio_train.csv',sep=";")


# In[ ]:


#Eliminar columna de id, que no es de utilidad
data=data.drop(columns=["id"])
intro()


# In[ ]:


#Definiendo "X" y "y" 
X=data.iloc[:,:-1]
y=data.cardio
show_variables()


# In[ ]:


#División del set de datos 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("\nDivisión del set de datos exitosa: 80% train - 20% test\n")
print("----------------------------------------------------------------------------------------\n")


# In[ ]:


#Creación del modelo
clf = RandomForestClassifier()
print("Creación de modelo \"Random Forest Classifier\" exitosa \n(Hiperparámetros configurados por default)\n") 


# In[ ]:


#Entrenamiento del modelo
clf.fit(X_train, y_train)
print("Entrenamiento del modelo exitoso")


# In[ ]:


#Hacer predicciones
predicciones=clf.predict(X_test)
print("\n Predicciones realizadas con éxito ")
print("Total de predicciones: ",len(predicciones),"\n")
print("----------------------------------------------------------------------------------------\n")
print("Ejemplo de predicción: \n")
print("Datos de la persona: \n")

x = random.randint(0,14000)
for i in range(11):
    print(X.columns[i],": ",X_test.iloc[x,i])
    
print("\nPredicción: ",predicciones[0],"\n")
print("(0-> Sin enfermedad cardiovascular, 1->Con enfermedad cardiovascular)")


# In[ ]:


#Calificar el modelo
accuracy_clf=accuracy_score(predicciones,y_test)
print("----------------------------------------------------------------------------------------\n")
print("Accuracy score para el modelo: ",accuracy_clf)
print("----------------------------------------------------------------------------------------\n")


# In[ ]:


#Entrenar con todos los datos 

#Creación del modelo
clf2 = RandomForestClassifier()
#Entrenamiento del modelo
clf2.fit(X,y)
print("Creación y entrenamiento de nuevo modelo ahora con todos los datos")
print("----------------------------------------------------------------------------------------\n")


# In[ ]:


print("Introduzca los siguientes datos para hacer una predicción: \n")
features=[]
for i in range(11):
    var=float(input((str(i+1)+"-> "+X.columns[i]+":  ")))
    features.append(var)


# In[ ]:


features


# In[ ]:


data_predict=np.array(features)
data_predict=data_predict.reshape(1,11)


# In[ ]:


prediccion=clf2.predict(data_predict)


# In[ ]:


print("\n La predicción para una persona con esas características es: ",prediccion,"\n")


# In[ ]:


final()

