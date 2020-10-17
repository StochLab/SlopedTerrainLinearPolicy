import numpy as np
import pybullet_data
import pybullet as p
import time as t
PI = np.pi
p.connect(p.GUI)
plane = p.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
p.changeVisualShape(plane,-1,rgbaColor=[1,1,1,0.9])
p.setGravity(0, 0, -9.8)
INIT_POSITION = [0,0,1]
stoch3 = p.loadURDF('gym_stoch2_sloped_terrain/envs/robots/stoch_4_urdf/urdf/stoch_4_urdf.urdf',[0,0,1])
laikago = p.loadURDF('gym_stoch2_sloped_terrain/envs/robots/laikago/laikago_toes_zup.urdf',[0,1,1])
p.createConstraint(stoch3, -1, -1, -1, p.JOINT_FIXED,[0, 0, 0], [0, 0, 0], [0, 0,INIT_POSITION[2]])
p.createConstraint(laikago, -1, -1, -1, p.JOINT_FIXED,[0, 0, 0], [0, 0, 0], [0, 1,INIT_POSITION[2]])
for j in range(p.getNumJoints(stoch3)):
    print("Joint_No:",j,p.getJointInfo(laikago,j),"\n")
while True:
    
    p.setJointMotorControlArray(stoch3,np.arange(12),p.POSITION_CONTROL,
                        targetPositions = [0,PI/4,-PI/8,
                                           0,-PI/4,-PI/8,
                                           0,PI/4,-PI/8,
                                           0,-PI/4,-PI/8])
    # 3,7,11,15
    p.setJointMotorControlArray(laikago,[0,1,2,4,5,6,8,9,10,12,13,14],p.POSITION_CONTROL,
                        targetPositions = [0,PI/4,-PI/8,
                                           0,PI/4,-PI/8,
                                           0,PI/4,-PI/8,
                                           0,PI/4,-PI/8])
    '''
    abduction : FL, BR one pair & FR & BL are one pair and the respective pairs are opposite in conventions
    hip : FL & BL are a pair & FR & BR are one pair and the respective pairs are opposite in conventions
    knee : the lower and upper limits of the joints are defines as 0 and 0, so it is not moving and is defined as a continous joint
    '''
    
    p.stepSimulation()
    t.sleep(1/240.0)