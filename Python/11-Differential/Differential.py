import numpy as np
import matplotlib.pylab as plt

#y = 0.5x^2 + 0.1x
#dy = 1x+0.1
#x = 5,y = 5.1
#x = 10,y = 10.1

def function(x):
	return 0.5*x**2 + 0.1*x
def function_differential(f,x):
	h = 1e-4 #0.0001 
	return (f(x+h) - f(x-h))/(2*h)
	
x = np.arange(0.0,20.0,0.1)
y = function(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

print(function_differential(function,5))
print(function_differential(function,10))