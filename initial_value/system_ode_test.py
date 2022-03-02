import numpy as np
import matplotlib.pyplot as plt
from methods import euler, rk4


def sample_func(u,t):
    x,v = u
    return [v ,9.81 - (0.25/68.1)*v**2]

def analytical_v(t):
    return np.sqrt(9.81*68.1/0.25) * np.tanh(np.sqrt(9.81*0.25/68.1)*t)

def analytical_x(t):
    return (68.1/0.25) * np.log(np.cosh(np.sqrt(9.81*0.25/68.1)*t))

initial_value = [0,0]
t = np.linspace(0,10,6)
x,v = euler(sample_func, initial_value,t)
x1,v1 = rk4(sample_func, initial_value,t)

plt.plot(t, analytical_x(t), label = "Analytical x")
plt.plot(t, analytical_v(t), label = "Analytical V")
plt.plot(t,x, label = "Euler x")
plt.plot(t,v, label = "Euler V")
plt.plot(t,x1, label = "rk4 x")
plt.plot(t,v1, label = "rk4 V")

plt.legend()
plt.show()


