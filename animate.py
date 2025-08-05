import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 初期状態の設定
w0 = [10.0,0.1,0.1] # ωの初期値
t_span = [0.0,5.0]
eta = 1.5

# 定数の設定
I = np.array([eta * eta, 1.0, eta * eta + 1.0])
q0 = [1.0, 0.0, 0.0, 0.0] # 初期の姿勢
init = np.concatenate((w0, q0))
fs = 150
t_eval = np.linspace(t_span[0], t_span[1], fs)

# 微分方程式の定義
def euler_motion(t, state):
    w = state[:3]
    q = state[3:]
    dw_dt = np.array([
        (I[1] - I[2]) / I[0] * w[1] * w[2],
        (I[2] - I[0]) / I[1] * w[2] * w[0],
        (I[0] - I[1]) / I[2] * w[0] * w[1]
    ])
    qw, qx, qy, qz = q
    omega_q = np.array([
        0.5 * (-w[0]*qx - w[1]*qy - w[2]*qz), 0.5 * ( w[0]*qw + w[2]*qy - w[1]*qz),
        0.5 * ( w[1]*qw - w[2]*qx + w[0]*qz), 0.5 * ( w[2]*qw + w[1]*qx - w[0]*qy)
    ])
    return np.concatenate((dw_dt, omega_q))

# 微分方程式を数値計算で解く
sol = solve_ivp(euler_motion, t_span, init, t_eval=t_eval, rtol=1e-9, atol=1e-12)

quaternions_wxyz = sol.y[3:].T
quaternions_xyz_w = quaternions_wxyz[:, [1, 2, 3, 0]]
rotations = Rotation.from_quat(quaternions_xyz_w)


# アニメーションの作成

# 長方形の頂点を定義
L_x, L_y, L_z = 1.0, eta, 0.1
verts = np.array([
    [-L_x/2, -L_y/2, -L_z/2], [L_x/2, -L_y/2, -L_z/2], [L_x/2, L_y/2, -L_z/2], [-L_x/2, L_y/2, -L_z/2],
    [-L_x/2, -L_y/2, L_z/2], [L_x/2, -L_y/2, L_z/2], [L_x/2, L_y/2, L_z/2], [-L_x/2, L_y/2, L_z/2]
])

# 面の定義
face_indices = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
]

# 描画のセットアップ
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# アニメーション更新関数
def update(frame):
    ax.cla()
    rotated_verts = rotations[frame].apply(verts) # 現在の回転を適用
    faces = [[rotated_verts[i] for i in face] for face in face_indices] # 面を構成する

    # 面を描画
    ax.add_collection3d(Poly3DCollection(
        faces,
        facecolors='cyan', 
        linewidths=1, 
        edgecolors='r',
        alpha=.25
    ))

    # 描画範囲、ラベルの設定
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Time: {t_eval[frame]:.2f}s')

intv = 1000 * t_span[1] / fs # シミュレーション内の1sが現実の1sに相当するように調整
# アニメーションを作成
ani = FuncAnimation(fig, update, frames=len(t_eval), interval=intv)

fps = fs / t_span[1]
print(fps)
ani.save("dzhanibekov.mp4",writer="ffmpeg",fps=fps)

# アニメーションを表示
# plt.show()
