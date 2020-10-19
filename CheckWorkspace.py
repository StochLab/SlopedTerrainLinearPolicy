import pybullet as p
import numpy as np
import math
import time 

p.connect(p.GUI)
model_path = 'gym_stoch2_sloped_terrain/envs/robots/laikago/laikago_toes_zup.urdf'
Laikago = p.loadURDF(model_path, [0,0,0.65])
prev_feet_points = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
def vis_foot_traj(line_thickness = 5,life_time = 15):
    LINK_ID = [3,7,11,15]
    i=0
    for  link_id in LINK_ID:
        current_point = p.getLinkState(Laikago,link_id)[0]
        p.addUserDebugLine(current_point,prev_feet_points[i],[1,0,0],line_thickness,lifeTime=life_time)
        prev_feet_points[i] = current_point
        i+=1

t = 0
step = 0
log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"wspace.mp4",)
last_angles = [0]*12
while True:
    t+=0.01
    step += 1

    p.resetBasePositionAndOrientation(Laikago,[0,0,0.65],[0,0,0,1])
    
    x =  0.5*math.cos(t)
    y = 0
    z = 0.1 + 0.5*math.sin(t)
    targets = [[x,y,z],
		       [x,y,z],
			   [x,y,z],
		       [x,y,z]]
    angles = p.calculateInverseKinematics2(Laikago,[3,7,11,15],targets)
    print(angles)
    p.setJointMotorControlArray(Laikago,[0,1,2,
                                         4,5,6,
                                         8,9,10,
                                         12,13,14],controlMode=p.POSITION_CONTROL,targetPositions=angles)#,currentPosition=last_angles)	
    p.stepSimulation()
    last_angles = angles
    if(step % 10 ==0):
        vis_foot_traj()
    time.sleep(0.01)
