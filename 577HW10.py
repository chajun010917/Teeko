import numpy as np
curr=[(2, 2)]
dis=0.0
npCurr = np.array(curr)
median = np.median(npCurr, axis=0)
for x in npCurr:
    dis+=np.linalg.norm(x-median)
print(dis)