
import pybullet as p
import pybullet_data
import os
import time
'''
urdf file in the same folder as that of the python script
'''
file_path = os.getcwd()

#Enter the file name with a '/' in front of it 
file_name = "wedge_"
'''
these comands are explained in detail in the next subpart
for now u can directly use it to visualize the model
'''
p.connect(p.GUI)
p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
for i in range(6):
	robot = p.loadURDF(file_name+str(5+2*i)+".urdf")
	p.resetBasePositionAndOrientation(robot, [2*i, 0, 0],p.getQuaternionFromEuler([0,0,-1.57]))
p.setGravity(0,0,-10)

while(True):
	time.sleep(0.01)
	p.stepSimulation()