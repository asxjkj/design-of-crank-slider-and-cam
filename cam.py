import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from scipy.optimize import minimize_scalar

#定义区间
x=np.linspace(0,2*pi,1000)
theta = np.linspace(0, 2 * np.pi, 1000)

#定义函数
def f1(x):
    return -18.09*cos(6/5*x)+18.09

def f2(x):
    return -8.85*cos(3*x)+27.33

def f3(x):
    return 9.24*sin(3*x)+9.24

def s(x):
    return np.piecewise(x, 
                        [ (0 <= x) & (x < 5/6*pi),
                          (5/6*pi <= x) & (x < pi),
                          (pi <= x) & (x < 4/3*pi),
                          (4/3*pi <= x) & (x < 3/2*pi),
                          (3/2*pi <= x) & (x < 11/6*pi),
                          (11/6*pi <= x) & (x <= 2*pi)],
                        [f1, 36.18, f2, 18.48, f3, 0])

def diff(func, x):
    dx = x[1] - x[0]
    return np.gradient(func(x), dx)

v = diff(s, x)
a = diff(lambda t: diff(s, t), x)

##求基圆半径
def f(x):
    return abs(v)*1.73205-s(x)

min= minimize_scalar(lambda t: f(np.array([t]))[0], bounds=(0, 2*pi), method='bounded')
print(f"最大值出现在 x = {min.x}")
print(f"最大值为 f(x) = {-min.fun}")
r0=-min.fun

y1=s(x)
y2=v
y3=a
r1=np.full_like(theta,r0)
r2=r0+s(x)

##绘制曲线图

#plt.plot(x*180/pi,f(x))
#plt.show()

plt.figure(figsize=(5,15))
plt.subplot(3,1,1)
plt.plot(x*180/pi,y1)
plt.ylabel('s(mm)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(x*180/pi,y2)
plt.ylabel('v(mm/°)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(x*180/pi,y3)
plt.ylabel('a(mm/°^2)')
plt.xlabel('delta(°)')
plt.grid(True)

plt.show()

##绘制凸轮

plt.polar(theta,r1,color='green')
plt.polar(theta,r2,color='red')
plt.grid(False)
plt.show()