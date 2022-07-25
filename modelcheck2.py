import numpy as np
import physics_sim
import matplotlib.pyplot as plt
import uavutils
env = physics_sim.PhysicsSim()
env.reset()
state=np.array([0,0,0,0,0,0,0,0,0,0,0,0])
speed=np.linspace(-1,1,200)
hahah = np.linspace(-1,1,200)
Fx = []
Fy = []
Fz = []
print(uavutils.alloaction_act(np.array([0,1,1,1])))
for i in range(200):
    actionx = np.array([speed[i],0,1,0])
    action = uavutils.alloaction_act(actionx)
    F,M=env.Calculate_Force(action[0],action[1:],state)
    Fx.append(F[0])
    Fy.append(F[1])
    Fz.append(F[2])
plt.subplot(3,1,1)
plt.plot(speed,np.array(Fx))
plt.subplot(3,1,2)
plt.plot(speed,np.array(Fy))
plt.subplot(3,1,3)
plt.plot(speed,np.array(Fz))
plt.show()
