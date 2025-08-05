import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from scipy.signal import find_peaks


def euler(t,w,lm):
    x,y,z = w
    dxdt = -y * z
    dydt = z * x
    dzdt = lm * x * y
    return np.array([dxdt,dydt,dzdt])

# w0とetaを与えることでひっくり返る周期を計算する
def get_span_sim(w0,eta,t_end=50.0,fs=1000):
    t_span = [0.0,t_end]
    lm = (eta * eta - 1) / (eta * eta + 1)
    t_eval = np.linspace(*t_span,fs) # time for sampling

    sol = solve_ivp(euler,t_span,w0,method='RK45',t_eval=t_eval,args=(lm,),rtol=1e-9,atol=1e-12)

    # xの角速度を見る
    sx = sol.y[0,:]

    # 極値を見つける
    p,_ = find_peaks(sx)
    pn = len(p)

    # 極値間の長さを測って、その平均を出す
    avs = sum(p[i + 1] - p[i] for i in range(pn - 1)) / (pn - 1)

    # 時間に変換する
    sp = avs * (t_end / fs)
    return sp

# 角速度のy,z成分の初期値たち
dws = np.arange(0.01,1.0,0.01)
sps = []

for i in dws:
    sp = get_span_sim([10,i,i],1.5)
    sps.append(sp)

sps = np.array(sps)
print(sps)

plt.plot(dws,sps)
plt.legend(["real"])
plt.show()
