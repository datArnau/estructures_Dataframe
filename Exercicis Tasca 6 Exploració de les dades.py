#!/usr/bin/env python
# coding: utf-8

# # Tasca 6 - Exploració de les dades

# ## Exercici 1

# In[22]:


import pandas as pd 

dataframe= pd.read_csv('C:/Users/TREBALL/Desktop/BCN-Activa-Data Science/database/DelayedFlights.csv')
df = pd.DataFrame(dataframe)

#display("El DataFrame :")
#display(df) 
#display("Les capçaleres de les columnes :")
#display(list(df.columns.values))
df_backup = df

del df['Year'] # he tret la columna d'any perque totes les dades pertanyen al mateix any i no es preveu ampliar la taula, per tant, aquesta dada no aprota informació rellevant
del df['DepTime'] # la dada està duplicada. He escollit la dada estandaritzada i he descartart la relativa
del df['ArrTime'] # la dada està duplicada. He escollit la dada estandaritzada i he descartart la relativa
del df['CancellationCode'] # Aquest codi no m'ha semblat rellevant perque tots els valors son 'N' 

display("El DataFrame modificat:")
display(df)

display("Les capçaleres de les columnes :")
display(list(df.columns.values))


# In[14]:


import numpy as np

print("Info genèrica de la taula: ")
print("-")
print(df.info())
print("-")
print("Nº de files i columnes de la taula: ")
print(df.shape) 
print("-")
print("Info estadística sobre les columnes de la taula: ")
print("-")
print("Columna Month :")
print("La moda es " + str(df["Month"].median()))
print("La mitjana es " + str(df["Month"].mean()))
print("La maxima es " + str(df["Month"].max()))
print("La minima es " + str(df["Month"].min()))
print("La la variança es " + str(df["Month"].var()))
print("La desviació estandard es " + str(df["Month"].std()))
print("-")
print("Columna DayofMonth :")
print("La moda es " + str(df["DayofMonth"].median()))
print("La mitjana es " + str(df["DayofMonth"].mean()))
print("La maxima es " + str(df["DayofMonth"].max()))
print("La minima es " + str(df["DayofMonth"].min()))
print("La la variança es " + str(df["DayofMonth"].var()))
print("La desviació estandard es " + str(df["DayofMonth"].std()))
print("-")
print("Columna DayOfWeek :")
print("La moda es " + str(df["DayOfWeek"].median()))
print("La mitjana es " + str(df["DayOfWeek"].mean()))
print("La maxima es " + str(df["DayOfWeek"].max()))
print("La minima es " + str(df["DayOfWeek"].min()))
print("La la variança es " + str(df["DayOfWeek"].var()))
print("La desviació estandard es " + str(df["DayOfWeek"].std()))
print("-")
print("Columna CRSDepTime :")
print("La moda es " + str(df["CRSDepTime"].median()))
print("La mitjana es " + str(df["CRSDepTime"].mean()))
print("La maxima es " + str(df["CRSDepTime"].max()))
print("La minima es " + str(df["CRSDepTime"].min()))
print("La la variança es " + str(df["CRSDepTime"].var()))
print("La desviació estandard es " + str(df["CRSDepTime"].std()))
print("-")
print("Columna CRSArrTime :")
print("La moda es " + str(df["CRSArrTime"].median()))
print("La mitjana es " + str(df["CRSArrTime"].mean()))
print("La maxima es " + str(df["CRSArrTime"].max()))
print("La minima es " + str(df["CRSArrTime"].min()))
print("La la variança es " + str(df["CRSArrTime"].var()))
print("La desviació estandard es " + str(df["CRSArrTime"].std()))
print("-")
print("Columna UniqueCarrier :")
print("-")
print("-")
print("Columna FlightNum :")
print("La moda es " + str(df["FlightNum"].median()))
print("La mitjana es " + str(df["FlightNum"].mean()))
print("La maxima es " + str(df["FlightNum"].max()))
print("La minima es " + str(df["FlightNum"].min()))
print("La la variança es " + str(df["FlightNum"].var()))
print("La desviació estandard es " + str(df["FlightNum"].std()))
print("-")
print("Columna ActualElapsedTime :")
print("La moda es " + str(df["ActualElapsedTime"].median()))
print("La mitjana es " + str(df["ActualElapsedTime"].mean()))
print("La maxima es " + str(df["ActualElapsedTime"].max()))
print("La minima es " + str(df["ActualElapsedTime"].min()))
print("La la variança es " + str(df["ActualElapsedTime"].var()))
print("La desviació estandard es " + str(df["ActualElapsedTime"].std()))
print("-")
print("Columna CRSElapsedTime :")
print("La moda es " + str(df["CRSElapsedTime"].median()))
print("La mitjana es " + str(df["CRSElapsedTime"].mean()))
print("La maxima es " + str(df["CRSElapsedTime"].max()))
print("La minima es " + str(df["CRSElapsedTime"].min()))
print("La la variança es " + str(df["CRSElapsedTime"].var()))
print("La desviació estandard es " + str(df["CRSElapsedTime"].std()))
print("-")
print("Columna AirTime :")
print("La moda es " + str(df["AirTime"].median()))
print("La mitjana es " + str(df["AirTime"].mean()))
print("La maxima es " + str(df["AirTime"].max()))
print("La minima es " + str(df["AirTime"].min()))
print("La la variança es " + str(df["AirTime"].var()))
print("La desviació estandard es " + str(df["AirTime"].std()))
print("-")
print("Columna ArrDelay :")
print("La moda es " + str(df["ArrDelay"].median()))
print("La mitjana es " + str(df["ArrDelay"].mean()))
print("La maxima es " + str(df["ArrDelay"].max()))
print("La minima es " + str(df["ArrDelay"].min()))
print("La la variança es " + str(df["ArrDelay"].var()))
print("La desviació estandard es " + str(df["ArrDelay"].std()))
print("-")
print("Columna DepDelay :")
print("La moda es " + str(df["DepDelay"].median()))
print("La mitjana es " + str(df["DepDelay"].mean()))
print("La maxima es " + str(df["DepDelay"].max()))
print("La minima es " + str(df["DepDelay"].min()))
print("La la variança es " + str(df["DepDelay"].var()))
print("La desviació estandard es " + str(df["DepDelay"].std()))
print("-")
print("Columna Origin :")
print("-")
print("-")
print("Columna Dest :")
print("-")
print("-")
print("Columna Distance :")
print("La moda es " + str(df["Distance"].median()))
print("La mitjana es " + str(df["Distance"].mean()))
print("La maxima es " + str(df["Distance"].max()))
print("La minima es " + str(df["Distance"].min()))
print("La la variança es " + str(df["Distance"].var()))
print("La desviació estandard es " + str(df["Distance"].std()))
print("-")
print("Columna TaxiIn :")
print("La moda es " + str(df["TaxiIn"].median()))
print("La mitjana es " + str(df["TaxiIn"].mean()))
print("La maxima es " + str(df["TaxiIn"].max()))
print("La minima es " + str(df["TaxiIn"].min()))
print("La la variança es " + str(df["TaxiIn"].var()))
print("La desviació estandard es " + str(df["TaxiIn"].std()))
print("-")
print("Columna TaxiOut :")
print("La moda es " + str(df["TaxiOut"].median()))
print("La mitjana es " + str(df["TaxiOut"].mean()))
print("La maxima es " + str(df["TaxiOut"].max()))
print("La minima es " + str(df["TaxiOut"].min()))
print("La la variança es " + str(df["TaxiOut"].var()))
print("La desviació estandard es " + str(df["TaxiOut"].std()))
print("-")
print("Columna Cancelled :")
print("La moda es " + str(df["Cancelled"].median()))
print("La mitjana es " + str(df["Cancelled"].mean()))
print("La maxima es " + str(df["Cancelled"].max()))
print("La minima es " + str(df["Cancelled"].min()))
print("La la variança es " + str(df["Cancelled"].var()))
print("La desviació estandard es " + str(df["Cancelled"].std()))
print("-")
print("Columna Diverted :")
print("La moda es " + str(df["Diverted"].median()))
print("La mitjana es " + str(df["Diverted"].mean()))
print("La maxima es " + str(df["Diverted"].max()))
print("La minima es " + str(df["Diverted"].min()))
print("La la variança es " + str(df["Diverted"].var()))
print("La desviació estandard es " + str(df["Diverted"].std()))
print("-")
print("Columna CarrierDelay :")
print("La moda es " + str(df["CarrierDelay"].median()))
print("La mitjana es " + str(df["CarrierDelay"].mean()))
print("La maxima es " + str(df["CarrierDelay"].max()))
print("La minima es " + str(df["CarrierDelay"].min()))
print("La la variança es " + str(df["CarrierDelay"].var()))
print("La desviació estandard es " + str(df["CarrierDelay"].std()))
print("-")
print("Columna WeatherDelay :")
print("La moda es " + str(df["WeatherDelay"].median()))
print("La mitjana es " + str(df["WeatherDelay"].mean()))
print("La maxima es " + str(df["WeatherDelay"].max()))
print("La minima es " + str(df["WeatherDelay"].min()))
print("La la variança es " + str(df["WeatherDelay"].var()))
print("La desviació estandard es " + str(df["WeatherDelay"].std()))
print("-")
print("Columna NASDelay :")
print("La moda es " + str(df["NASDelay"].median()))
print("La mitjana es " + str(df["NASDelay"].mean()))
print("La maxima es " + str(df["NASDelay"].max()))
print("La minima es " + str(df["NASDelay"].min()))
print("La la variança es " + str(df["NASDelay"].var()))
print("La desviació estandard es " + str(df["NASDelay"].std()))
print("-")
print("Columna SecurityDelay :")
print("La moda es " + str(df["SecurityDelay"].median()))
print("La mitjana es " + str(df["SecurityDelay"].mean()))
print("La maxima es " + str(df["SecurityDelay"].max()))
print("La minima es " + str(df["SecurityDelay"].min()))
print("La la variança es " + str(df["SecurityDelay"].var()))
print("La desviació estandard es " + str(df["SecurityDelay"].std()))
print("-")
print("Columna LateAircraftDelay :")
print("La moda es " + str(df["LateAircraftDelay"].median()))
print("La mitjana es " + str(df["LateAircraftDelay"].mean()))
print("La maxima es " + str(df["LateAircraftDelay"].max()))
print("La minima es " + str(df["LateAircraftDelay"].min()))
print("La la variança es " + str(df["LateAircraftDelay"].var()))
print("La desviació estandard es " + str(df["LateAircraftDelay"].std()))


# In[15]:


print("Recompte de valors NaN per columna:")

print(df.isna().sum())
        


# In[26]:


import numpy as np

# Velocitat mitjana dels vols

vel_mitjana = df['Distance'] / df['AirTime']
df['Velocitat_mitjana'] = vel_mitjana

# Vols que han arribat tard (1 = Arriba tard, 2 = No arriba tard)

df["Arriba_Tard"] = np.where(df["ArrDelay"]>0,1,0)

# Trajectes

df['Trajecte']=df['Origin'].astype(str)+' - '+df['Dest'].astype(str)

# Mostra les noves columnes

y = df[df.columns[-3:]]
y


# In[39]:


import matplotlib.pyplot as plt 
import seaborn as sns 

print("Rutes de més distància:")

plot_order = df.groupby('Trajecte')['Distance'].sum().sort_values(ascending=False).index.values

Aerol_delay = df[['Trajecte', 'Distance']]
Aerol_delay = Aerol_delay.groupby(by='Trajecte').sum()
Aerol_delay=Aerol_delay.reset_index(drop=False)
Aerol_delay.head(7)
plt.figure(figsize=(15,8))
sns.barplot(x="Trajecte", y="Distance", data=Aerol_delay, order=plot_order[1:15])
plt.show(10)


# In[71]:


print("Aerolinies que més endarreriments tenen:")
Aerol_delay = df[['UniqueCarrier', 'DepDelay']]
Aerol_delay = Aerol_delay.groupby(by='UniqueCarrier').sum()
Aerol_delay=Aerol_delay.reset_index(drop=False)
Aerol_delay.head(7)
plt.figure(figsize=(15,8))
sns.barplot(x="UniqueCarrier", y="DepDelay", data=Aerol_delay,order=['WN', 'AA', 'UA', 'MQ','OO','XE','CO','DL','EV','YV',
                                                                                'US', 'NW','FL', 'B6','OH','9E',
                                                                                 'AS','F9','HA','AQ'])
plt.show()


# In[99]:


print("Aerolinies que més endarreriments tenen:")

nou_df = df[['UniqueCarrier', 'DepDelay']]

na = nou_df.groupby(by='UniqueCarrier', sort=False).sum()
na.sort_values(by=['DepDelay'], ascending=False)


# In[40]:


print("Rutes que més endarreriments tenen:")

plot_order = df.groupby('Trajecte')['DepDelay'].sum().sort_values(ascending=False).index.values

Aerol_delay = df[['Trajecte', 'DepDelay']]
Aerol_delay = Aerol_delay.groupby(by='Trajecte').sum()
Aerol_delay=Aerol_delay.reset_index(drop=False)
Aerol_delay.head(7)
plt.figure(figsize=(15,8))
sns.barplot(x="Trajecte", y="DepDelay", data=Aerol_delay, order=plot_order[1:15])
plt.show(10)


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns
print("Matriu de Correlació:")
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);
plt.show()


# In[13]:


# Valoro la possiblitat d'interès a canviar la columna "0" (valors progressius per cada registre)...
# per la columna TailNum, ja que aquesta conté valors únics per cada vol


# In[44]:


# El Data Set és massa gran per a expoortarlo a Excel. L'exporto a CSV
df.to_csv(r'C:\Users\TREBALL\Desktop\BCN-Activa-Data Science\database\Exercicis_Exploracio_Dades.csv', index = False)

