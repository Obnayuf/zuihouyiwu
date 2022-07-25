import numpy as np
import matplotlib.pyplot as plt
import uavutils
import physics_sim
env=physics_sim.PhysicsSim()

env.reset()
while not env.done:
    action=env.action_space.sample()
    s_prime, r, done, info = env.step(action)
uavutils.save_figure(env.time,env.stepid,env.statebuffer,2,1)


