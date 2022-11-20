import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = pd.read_excel("D:\\coding\\isro internship\\sowmiya_ph.xlsx")
initial_pH = file["pH"]
date = file["Date"]
sliding_pH = []
sliding_value, temp = 1000, 0
for i in range(sliding_value):
    temp += initial_pH[i]
temp /= sliding_value
sliding_pH.append(temp)
for i in range(1, len(initial_pH)):
    temp = 0
    if i+sliding_value > len(initial_pH):
        break
    c = 0
    while c < sliding_value:
        temp2 = i+c
        temp += initial_pH[temp2]
        c+=1
    temp /= sliding_value
    sliding_pH.append(temp)
diff = len(initial_pH) - len(sliding_pH)
for i in range(diff):
    sliding_pH.append(np.nan)
file.insert(9, "SpH", sliding_pH)
#print(file.head(5))
plt.plot(date,initial_pH,date,sliding_pH,"--k")
plt.title("pH Graph")
plt.ylabel("pH")
plt.legend(['Initial pH','Sliding pH'])
plt.show()
