import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
#buliding data
x = np.arange(0,6,0.1)#form 0 to 6,unit 0.1
y1 = np.sin(x)
y2 = np.cos(x)
#drawing 
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle="--",label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')
plt.legend()
#plt.show()

img = imread('th.jpg')
plt.imshow(img)
plt.show()