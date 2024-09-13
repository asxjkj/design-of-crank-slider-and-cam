import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from scipy.optimize import fsolve
from scipy.misc import derivative

##求解曲柄长和连杆长
H = float(input("滑块行程H(mm): "))
e = float(input("偏心距e(mm): "))
A_deg = float(input("极位夹角theta(°): "))
A_rad = math.radians(A_deg) #极位夹角(rad)

def equations(vars, H, e, A_rad):
    b, c = vars
    eq1 = H * e - b * c * sin(A_rad)
    eq2 = H**2 - (b**2 + c**2 - 2 * b * c * cos(A_rad))
    return [eq1, eq2]

initial_guess = [1, 1]
b, c = fsolve(equations, initial_guess, args=(H, e, A_rad))

l=(b+c)/2 #连杆长
r=abs(b-c)/2 #曲柄长
alpha=math.degrees(pi/2-math.acos((e+r)/l)) #最大压力角

print(f"连杆长度为: {l:.2f}mm")
print(f"曲柄长度为: {r:.2f}mm")
print(f"最大压力角为: {alpha:.2f}°")

##绘制图表

rpm = float(input("曲柄转速(r/min):"))
omega = (2*pi*rpm)/60 #曲柄转速(rad/s)

#输入时间范围
t0 = float(input("机构运动秒数:"))
time = np.linspace(0,t0,1000)

def disp(t): #关于时间的滑块位移函数
    d1 = r*cos(omega*t)
    d2 = np.sqrt((l**2-(e-r*sin(omega*t))**2))
    d3=np.sqrt(((l+r)**2-e**2))
    return d3-d1-d2

def vel(t): #关于时间的滑块速度函数
    return -derivative(disp,t)

def acc(t): #关于时间的滑块加速度函数
    return -derivative(vel,t)

x = np.array([disp(ti) for ti in time])
v = np.array([vel(ti) for ti in time])
a = np.array([acc(ti) for ti in time])
theta = omega*time*180/pi #曲柄角度(°)

#绘制曲线图
plt.figure(figsize=(5,15)) #新建画布

plt.subplot(3,1,1)
plt.plot(theta,x,color="green")
plt.title("Graphs of a slider-crank mechanism")
plt.ylabel("displacement(mm)")
plt.grid(color="blue")

plt.subplot(3,1,2)
plt.plot(theta,v,color="orange")
plt.ylabel("velocity(mm/s)")
plt.grid(color="blue")

plt.subplot(3,1,3)
plt.plot(theta,a,color="red")
plt.xlabel("angle(°)")
plt.ylabel("acceleration(mm/s^2)")
plt.grid(color="blue")

plt.show() #显示图表