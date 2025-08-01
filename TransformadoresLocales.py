import pandas as pd
## Importando estimadores, transformadores, clasificadores y regresores.
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# TEMPLATES PARA RECORDAR COMO ESCRIBIR LAS CLASES NECESARIAS
default_value = None

# TRANSFORMADORES ###################################################
class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=default_value):
        self.param1 = param1
        
    def fit(self, X, y=None):
        # Lógica de aprendizaje (ej: calcular medias)
        return self  # Debe retornar self
    
    def transform(self, X):
        # Aplica transformación (ej: escalado personalizado)
        X_transformed = ...  # Implementa tu lógica aquí
        return X_transformed

# CLASIFICADORES ###################################################
class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, param1=default_value):
        self.param1 = param1
        
    def fit(self, X, y):
        # Entrena el modelo (ej: optimizar parámetros)
        self.model_ = ...  # Almacena el modelo entrenado
        return self
    
    def predict(self, X):
        # Retorna predicciones categóricas
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        # Retorna probabilidades (opcional)
        return self.model_.predict_proba(X)

# REGRESORES ########################################################
class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, param1=default_value):
        self.param1 = param1
        
    def fit(self, X, y):
        # Implementa algoritmo físico-matemático aquí
        self.coef_ = ...  # Guarda coeficientes
        return self
    
    def predict(self, X):
        # Retorna predicciones numéricas
        return X @ self.coef_
    
###################################################################################
###################################################################################

# Transformador de PassengerId
class VolverADataFrame_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        nombres = [
                    #'CryoSleep', 'VIP',  # vienen tal cual
                    'GrupoId', 'GrupoSize', 'Solo', # salen de passengerID  
                    'T-T', 'A_trappist', 'terricola', # salen de HomePlanet y Destination 
                    'Piso', 'N_cabina', 'Lado', 'Zona', # Salen de Cabin P/N/L
                   'Rango etario', 'MenorDeEdad', # categorizacion de la edad
                   'NFAbordo', # Sale de contar los apellidos iguales
                   'Gasto Total', 'GastoMayor', # Sale de los gastos del pasajero
                ]
        
        final = pd.DataFrame(X, columns=nombres)

        return final

# Transformador de PassengerId
class PassengerId_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data : pd.DataFrame = X.copy()
        data[['GrupoId','subId']] = data['PassengerId'].str.split('_',expand=True ).astype(int)
        data['GrupoSize'] = data[['GrupoId','subId']].groupby('GrupoId')['GrupoId'].transform('count')
        data['Solo'] = (data['GrupoSize']== 1).astype(int)
        
        
        return data.drop(['PassengerId','subId'], axis=1)

# Transformador de Homeplanet y Destination
class Itinerario_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        data['T-T'] = (data['HomePlanet'] == 'Earth') & (data['Destination'] == 'TRAPPIST-1e')
        data['A_trappist'] = (data['Destination'] == 'TRAPPIST-1e')
        data['terricola'] = (data['HomePlanet'] == 'Earth')

        return data.drop(['HomePlanet', 'Destination'], axis=1)
    
class Cabin_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        

        data[['Piso', 'N_cabina', 'Lado']] = data['Cabin'].str.split('/', expand=True)
        data['N_cabina'] = data['N_cabina'].astype(int)

        data['Zona'] = ''
        data.loc[  data['N_cabina'] <= 300, 'Zona' ] = 'Zona 1'
        data.loc[  (data['N_cabina'] > 300) & (data['N_cabina'] <= 700), 'Zona' ] = 'Zona 2'
        data.loc[  (data['N_cabina'] > 700) & (data['N_cabina'] <= 1175), 'Zona' ] = 'Zona 3'
        data.loc[  (data['N_cabina'] > 1175) , 'Zona' ] = 'Zona 4'
        
        return data.drop('Cabin', axis=1)
    
class Age_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        data['Rango etario'] = ''
        data.loc[  data['Age'] <= 18, 'Rango etario' ] = 'Menor'
        data.loc[  (data['Age'] > 18) & (data['Age'] <= 30), 'Rango etario' ] = 'Adulto Joven'
        data.loc[  (data['Age'] > 30) & (data['Age'] <= 50), 'Rango etario' ] = 'Adulto'
        data.loc[  (data['Age'] > 50 ), 'Rango etario' ] = 'Adulto Mayor'

        data['MenorDeEdad'] = data['Age'] <= 18 

        return data.drop('Age', axis=1)
    
class Name_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data : pd.DataFrame = X.copy()

        data[['Nombre', 'Apellido']] = data['Name'].str.split(' ', expand=True)
        apellido_count = data['Apellido'].value_counts()
        data['NFAbordo'] = data['Apellido'].map(apellido_count)


        # si no tiene documentos -> vino solo
        data['NFAbordo']= data['NFAbordo'].fillna(0)

        data['NFAbordo'] = data['NFAbordo'].astype(int)

        

        return data.drop(['Name', 'Nombre', 'Apellido'], axis=1,)
    
class Gastos_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data : pd.DataFrame = X.copy()
        label_gastos = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        data['Gasto Total'] = data[label_gastos].sum(axis=1,skipna=False)#.astype(int)
        data['GastoMayor'] = data[  'Gasto Total'  ] < 100

        data = data.drop(label_gastos+['CryoSleep', 'Age'], axis=1)

        return data

class Imputer_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data : pd.DataFrame = X.copy()
        label_gastos = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        # Agregamos el itinerario Desconocido
        data['HomePlanet'] = data['HomePlanet'].fillna('Desconocido')
        data['Destination'] = data['Destination'].fillna('Desconocido')

        # Rellenamos con una nueva categoría para piso y lado. para el número, seleccionamos uno perteneciente al grupo con mas ocurrencias
        data['Cabin'] = data['Cabin'].fillna('X/0000/X')

        # La edad la rellenamos con la mediana en caso de haber outliers.
        data['Age'] = data['Age'].fillna( data['Age'].median() )

        # Manejo de valores nulos nombre
        # no es necesario ya que el transformador cuenta solo la cantidad de familiares
        # pasajeros sin nombre se cuentan como sin familiares.
    

        # Manejo cryosleep
        gasto_total = data[label_gastos].sum(axis=1,skipna=False)
        # donde gasto total es 0 y haya un nulo en cryosleep rellenar con true
        # ya que si no gasto nada puede que haya estado dormido
        data.loc[ gasto_total == 0 & data['CryoSleep'].isnull(), 'CryoSleep' ] = True
        data.loc[ gasto_total != 0 & data['CryoSleep'].isnull(), 'CryoSleep' ] = False

        # Para el vip cualquiera que haya gastado mas del 95% que el resto
        vip_threshold =gasto_total.quantile(0.95)

        #Imputar VIP=True para alto gasto
        data.loc[(gasto_total >=vip_threshold) & data['VIP'].isnull(), 'VIP'] = True
        data.loc[(gasto_total < vip_threshold) & data['VIP'].isnull(), 'VIP'] = False

        data['VIP'] = data['VIP'].astype(bool)

        # Los gastos rellenamos por partes.
        
        # Se le asigna 0 a todos los pasajeros dormidos y a los mejores de 15 años
        data.loc[data['CryoSleep'] == True, label_gastos] = 0
        data.loc[ data['Age'] <= 15 , label_gastos] = 0
        # El resto se rellena con la mediana
        for col in label_gastos:
            data[col] = data[col].fillna(data[col].median())
        
        return data

        