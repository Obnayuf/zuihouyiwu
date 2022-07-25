import math
from math import atan2

import numpy as np
from scipy.interpolate import interp1d
from gym import spaces

eps = 1e-10


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def body_to_earth_frame(PHI, THETA, PSI):  # 机体到地球的旋转坐标系
    # C^b_n
    # R = [[C(PSI) * C(THETA) - S(PSI) * S(PHI) * S(THETA), S(PSI) * C(PHI),
    #       C(PSI) * S(THETA) + S(PSI) * S(PHI) * C(THETA)],
    #      [S(PSI) * C(THETA) + C(PSI) * S(PHI) * S(THETA), C(PSI) * C(PHI),
    #       S(PHI) * S(THETA) - C(PSI) * S(PHI) * C(THETA)],
    #      [-C(PHI) * S(THETA), S(PHI), C(PHI) * C(THETA)]]
    R = [[C(PSI) * C(THETA), C(PSI) * S(THETA) * S(PHI) - S(PSI) * C(PHI),
          C(PSI) * S(THETA) * C(PHI) + S(PSI) * S(PHI)],
         [C(THETA) * S(PSI), S(PSI) * S(THETA) * S(PHI) + C(PSI) * C(PHI),
          S(PSI) * S(THETA) * C(PHI) - C(PSI) * S(PHI)],
         [-S(THETA), S(PHI) * C(THETA), C(PHI) * C(THETA)]]
    return np.array(R)


def euler_angular_v(PHI, THETA, PSI):  # 机体欧拉角速率与角速度的关系
    Q = [[1, S(PHI) * np.tan(THETA), np.tan(THETA) * C(PHI)],
         [0, C(PHI), -S(PHI)],
         [0, S(PHI) / C(THETA), C(PHI) / C(THETA)]]
    return np.array(Q)


def earth_to_body_frame(PHI, THETA, PSI):  # 地球到机体
    # C^n_b
    return np.transpose(body_to_earth_frame(PHI, THETA, PSI))


def Constrain(x, x_min, x_max):
    if x > x_max:
        return x_max
    elif x < x_min:
        return x_min
    else:
        return x


def angle_correct(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def calculate_alpha(Amplitude, Vc):
    if Amplitude < 1.0e-05:
        alpha = 0.0
    else:
        alpha = np.arccos(Vc / Amplitude)
    if alpha > 3.141:
        alpha = 3.14
    elif alpha < 0:
        alpha = 0.0
    return alpha


class PhysicsSim(object):
    '''单涵道无人机垂直起降物理模型，参照了Design and experimental validation of a nonlinear control law for a ducted-fan miniature aerial vehicle
    来自2010 Control Engineering Practice'''

    def __init__(self):
        # mass and inertia
        self.I = np.array([25, 30, 28])  # 转动惯量单位kg*m^2
        self.mass = 100  # 单位 kg
        # duct
        self.Cduct = 0.2 # 单位 m^2
        self.r = 0.6 # 单位 m
        # Fuselage aerodynamics
        self.Clmax = 1.2
        self.Cla = 5
        self.Swing = 2.5
        self.Cymax = 0.5
        self.CyB = 3
        self.Sside = 0.5
        self.Cdgain_1 = 1.9
        self.Cdgain_2 = 0.5
        self.Cd0 = 0.04
        self.z1 = 0.05
        self.z2 = 0.06
        # control surfaces
        self.Lr = 0.3
        self.La = 0.3
        self.Le = 0.25
        self.Scs = 0.06
        # rotor
        self.b = 0.0045
        self.k = 0.0015
        self.wr = 260
        self.nb = 4
        self.ib = 0.0027
        # Miscellaneous
        self.p = 1.225
        self.g = 9.8
        self.L1 = 80

    def get_earth_V(self):
        return np.matmul(body_to_earth_frame(*list(self.pose[3:])), self.vb)

    def get_body_gravity(self):
        return np.matmul(earth_to_body_frame(*list(self.pose[3:])), np.array([0, 0, self.mass * self.g]))

    def reset(self):
        self.time = 0.0
        self.pose = np.array([0.0, 0.0, 25.0, 0.0, 0.0, 0.0])
        self.vb = np.array([0.0, 0.0, 0.5])  # 初始速度 机体坐标系
        self.angular_v = np.array([0.0, 0.0, 0.0])  # 初始角速度 w=p,q,r
        self.done = False
        self.stepid = 0
        self.already_crash = False
        self.already_landing = False
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []
        self.shaping = None
        obs = self.get_obs()
        return obs

    def test_model(self, action):
        T, a, b, c = action
        u, v, w = self.vb
        p, q, _ = self.angular_v
        Vi = np.sqrt(T / (2 * self.denisty * self.Sdisk))
        Fx = 0.5 * self.denisty * self.cL * self.S2 * a * (Vi ** 2)
        Fy = 0.5 * self.denisty * self.cL * self.S2 * b * (Vi ** 2)
        Fz = -T
        Frd = -self.denisty * self.Sdisk * Vi * np.array([u, v, 0])
        F = np.array([Fx, Fy, Fz]) + Frd + self.get_body_gravity()
        N = self.KN * T
        Nc = 0.5 * self.denisty * self.cL * self.S1 * (Vi ** 2) * c * (self.dT / 2)
        M = np.array([-Fy * self.d, Fx * self.d, N + Nc])
        M += self.Irot * np.sqrt(T / self.KT) * np.array([-q, p, 0])

    def get_obs(self):
        pass
