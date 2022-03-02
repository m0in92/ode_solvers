import numpy as np


def initialize_y(y0, x):
    if not isinstance(y0, list):
        y0 = [y0]
    y = np.zeros([len(y0),len(x)])
    y[:, 0] = np.array(y0)
    return y

def euler(func, y0, x):
    """
    Euler method for initial value ODE problems.
    :param func: function instance.
    :param y0: initial value.
    :param x: x values in list or numpy array.
    :return: the numpy array of y values.
    """
    y = initialize_y(y0, x)
    for i in range(1,len(x)):
        dx = x[i] - x[i-1]
        increment = np.array(func(y[:,i-1],x[i-1]))
        y[:,i] = y[:,i-1] + increment * dx
    return y

def heun(func, y0, x, iteration = 1):
    """
    Heun method for initial value ODE problems.
    :param func: function instance.
    :param y0: initial value.
    :param x: x values in list or numpy array.
    :param iteration: The number of iterations. Default set to 1
    :return: the numpy array of y values and the convergence error in percent.
    """
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        y_pred = y[i - 1] + func(y[i - 1], x[i - 1]) * dx
        for j in range(iteration):
            y[i] = y[i - 1] + (func(y[i - 1], x[i - 1]) + (func(y_pred, x[i]))) * dx / 2
            ea = abs((y[i] - y_pred) / (y[i])) * 100
            y_pred = y[i]
    return y, ea

def rk2(func, y0, x, a2 = 1/2):
    """
    Second-Order Runge-Kutta method for initial value ODE problems.
    :param func: function instance.
    :param y0: initial value.
    :param x: x values in list or numpy
    :param a2: value of a2 (1/2 for Heun's method; 1 for Midpoint method; 2/3 for Ralston's method)
    :return:
    """
    y = np.zeros(len(x))
    a1 = 1 - a2
    p1 = q11 = 1 / (2*a2)
    y[0] = y0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        k1 = func(y[i-1], x[i-1])
        k2 = func(y[i-1] + q11 * k1 * dx, x[i-1] + p1 * dx)
        y[i] = y[i-1] + (a1 * k1 + a2*k2) * dx
    return y

def rk4(func, y0, x):
    y = initialize_y(y0, x)
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        k1 = np.array(func(y[:,i-1], x[i-1]))
        k2 = np.array(func(y[:,i-1] + (1/2) * k1 * dx, x[i-1] + (1/2) * dx))
        k3 = np.array(func(y[:,i-1] + (1 / 2) * k2 * dx, x[i-1] + (1 / 2) * dx))
        k4 = np.array(func(y[:,i-1] + k3 * dx, x[i-1] + dx))
        y[:,i] = y[:,i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dx
    return y

def rk5(func, y0, x):
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        k1 = func(y[i-1], x[i-1])
        k2 = func(y[i-1] + (1/4) * k1 * dx, x[i-1] + (1/4) * dx)
        k3 = func(y[i-1] + (1/8)*k1*dx + (1/8)*k2*dx, x[i-1] + (1/4)*dx)
        k4 = func(y[i-1] - (1/2)*k2*dx + k3*dx, x[i-1] + (1/2) * dx)
        k5 = func(y[i-1] + (3/16)*k3*dx + (9/16)*k4*dx, x[i-1] + (3/4)*dx)
        k6 = func(y[i-1] - (3/7)*k1*dx + (2/7)*k2*dx + (12/7)*k3*dx - (12/7)*k4*dx + (8/7)*k5*dx, x[i-1] + dx)
        y[i] = y[i-1] + (1/90) * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6) * dx
    return y