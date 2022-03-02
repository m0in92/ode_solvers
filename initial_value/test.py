from methods import euler, heun, rk2, rk4, rk5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def sample_func(x,t):
    return 4 * np.exp(0.8*t) - 0.5 * x

def func(t):
    return (4/1.3)*(np.exp(0.8*t) - np.exp(-0.5*t)) + 2*np.exp(-0.5*t)


initial_value = 2
t = np.linspace(0,4,5)
analytical_soln = func(t)
euler_soln = euler(sample_func, initial_value, t)[0]
heun_soln = heun(sample_func, initial_value, t)[0]
heun_soln2, ea = heun(sample_func, initial_value, t, iteration=10)
rk2_soln = rk2(sample_func, initial_value, t, a2 = 1/2)
rk4_soln = rk4(sample_func, initial_value, t)[0]
rk5_soln = rk5(sample_func, initial_value, t)
# print("euler error", abs(euler_soln[1] - analytical_soln[1])/euler_soln[1])
print("heun error - iter 1", abs(heun_soln[1] - analytical_soln[1])/heun_soln[1])
print("heun error - iter 10", abs(heun_soln2[1] - analytical_soln[1])/heun_soln2[1], ea)
print("rk2", abs(rk2_soln[1] - analytical_soln[1])/rk2_soln[1])
print("rk4", abs(rk4_soln[1] - analytical_soln[1])/rk4_soln[1])
print("rk5", abs(rk5_soln[1] - analytical_soln[1])/rk5_soln[1])

#plot
# set the font globally
plt.rcParams.update({'font.family':'Arial'})
fig = plt.figure()
no_of_figures = 1
ax1 = fig.add_subplot(no_of_figures,1,1)
ax1.plot(t, analytical_soln, label = "Analytical")
ax1.plot(t, euler_soln, label = "Euler")
# ax1.plot(t, heun_soln, label = "Heun-iter1")
# ax1.plot(t, heun_soln2, label = "Heun-iter:10")
# ax1.plot(t, rk2_soln, label = "rk2")
ax1.plot(t, rk4_soln, label = "rk4")
ax1.plot(t, rk5_soln, label = "rk5")
plt.legend()
plt.show()