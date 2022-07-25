import numpy as np
import matplotlib.pyplot as plt
import uavutils
import physics_sim
env=physics_sim.PhysicsSim()
t=0
reward = 0
buffer=[]
for i in range(1):
    env.reset()
    while not env.done:
        action= np.random.rand(4)
        s_prime, r, done, info = env.step(action)
        buffer.append(r)
        print(r)
        reward += r
    t += env.stepid
label = ['Ub', 'Vb', 'Wb', 'p', 'q', 'r', 'Xe', 'Ye', 'Ze', 'PITCH', 'ROLL', 'YAW','Vx','Vy' ,'Vz','r']
t = np.linspace(0, env.time + 0.02, env.stepid)
data = list(map(list, zip(*env.state_buffer)))
data.append(buffer)
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.plot(t, data[i])
    plt.title(label[i])
plt.tight_layout()
plt.show()


