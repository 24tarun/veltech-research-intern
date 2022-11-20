import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
x=np.array([2,3,4,5,6,7,8],dtype=float)
y=np.array([10,15,20,25,30,35,40],dtype=float)
#df=pd.DataFrame
model.fit(x, y, epochs=200)
result=model.predict([10])
print(result)