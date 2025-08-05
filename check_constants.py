import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 問題の設定
w0 = [10.0,0.1,0.1] # ωの初期値
t_span = [0.0,50.0]
eta = 1.5

# 各種定数の設定
lm = (eta * eta - 1) / (eta * eta + 1)
I = [eta * eta,1.0,eta * eta + 1.0]
fs = 1000
t_eval = np.linspace(*t_span,fs)

# 微分方程式の定義
def euler(t,w):
    x,y,z = w
    dxdt = -y * z
    dydt = z * x
    dzdt = lm * x * y
    return np.array([dxdt,dydt,dzdt])

# 微分方程式を解く
sol = solve_ivp(euler,t_span,w0,method='RK45',t_eval=t_eval,rtol=1e-9,atol=1e-12)

# ωの変化のグラフ
wx = sol.y[0,:]
wy = sol.y[1,:]
wz = sol.y[2,:]
plt.plot(sol.t,wx)
plt.plot(sol.t,wy)
plt.plot(sol.t,wz)
plt.legend(['wx','wy','wz'])
plt.savefig("w.png")


# 保存則の確認 (1)E
plt.clf()
E = (wx * wx * I[0] + wy * wy * I[1] + wz * wz * I[2]) / 2
plt.plot(sol.t,E)
plt.legend(['E'])
plt.savefig("E.png")
print("Error of K.E.:",(max(E) - min(E)) / min(E))

# 保存則の確認 (2)|L|
plt.clf()
L2 = wx * wx * I[0] * I[0] + wy * wy * I[1] * I[1] + wz * wz * I[2] * I[2]
plt.plot(sol.t,L2)
plt.legend(['L^2'])
plt.savefig("L2.png")
print("Error of L^2:",(max(L2) - min(L2)) / min(L2))
