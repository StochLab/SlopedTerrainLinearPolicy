# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
import os
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
PI = np.pi
no_of_points=100

@dataclass
class leg_data:
    name : str
    motor_hip : float = 0.0
    motor_knee : float = 0.0
    motor_abduction : float = 0.0
    x : float = 0.0
    y : float = 0.0
    radius : float = 0.0
    theta : float = 0.0
    phi : float = 0.0
    gamma : float = 0.0
    b: float = 1.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0
@dataclass
class robot_data:
    front_right : leg_data = leg_data('fr')
    front_left : leg_data = leg_data('fl')
    back_right : leg_data = leg_data('br')
    back_left : leg_data = leg_data('bl')

class WalkingController():
    def __init__(self,
                 gait_type='trot',
                 leg = [0.12,0.15015,0.04,0.15501,0.11187,0.04,0.2532,2.803],
                 phase = [0,0,0,0],
                 comy = 0.0
                 ):
        ## These are empirical parameters configured to get the right controller, these were obtained from previous training iterations  
        self._phase = robot_data(front_right = phase[0], front_left = phase[1], back_right = phase[2], back_left = phase[3])
        self.front_left = leg_data('fl')
        self.front_right = leg_data('fr')
        self.back_left = leg_data('bl')
        self.back_right = leg_data('br')
        self.gait_type = gait_type
        self._leg = leg
        self.gamma = 0
        self.MOTOROFFSETS = [2.3562,1.2217]
        self.body_width = 0.24
        self.body_length = 0.37

        #Trapezium up down limit [ -0.145 , -0.24] left right limit []
        self._pts = np.array([[-0.068,-0.24],[-0.115,-0.24],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.24],[0.068,-0.24]])
        # self.new_pts = np.array([[-0.058,-0.24],[-0.105,-0.24],[-0.055,-0.145],[0.075,-0.145],[0.125,-0.24],[0.078,-0.24]])
        self.comy = comy

    def update_leg_theta(self,theta):
        """ Depending on the gait, the theta for every leg is calculated"""
        def constrain_theta(theta):
            theta = np.fmod(theta, 2*no_of_points)
            if(theta < 0):
                theta = theta + 2*no_of_points
            return theta
        self.front_right.theta = constrain_theta(theta+self._phase.front_right)
        self.front_left.theta = constrain_theta(theta+self._phase.front_left)
        self.back_right.theta = constrain_theta(theta+self._phase.back_right)
        self.back_left.theta = constrain_theta(theta+self._phase.back_left)


    def initialize_elipse_shift(self, Yshift, Xshift, Zshift):
        self.front_right.y_shift = Yshift[0]
        self.front_left.y_shift = Yshift[1]
        self.back_right.y_shift = Yshift[2]
        self.back_left.y_shift = Yshift[3]

        self.front_right.x_shift = Xshift[0]
        self.front_left.x_shift = Xshift[1]
        self.back_right.x_shift = Xshift[2]
        self.back_left.x_shift = Xshift[3]

        self.front_right.z_shift = Zshift[0]
        self.front_left.z_shift = Zshift[1]
        self.back_right.z_shift = Zshift[2]
        self.back_left.z_shift = Zshift[3]

    def initialize_leg_state(self, theta, action):
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)

        self.update_leg_theta(theta)

        leg_sl = action[:4]  # fr fl br bl
        leg_phi = action[4:8]  # fr fl br bl

        self._update_leg_phi_val(leg_phi)
        self._update_leg_step_length_val(leg_sl)

        self.initialize_elipse_shift(action[8:12], action[12:16], action[16:20])

        return legs
    
    
    def run_eliptical_Traj(self, theta, action):

        legs = self.initialize_leg_state(theta, action)

        y_center = -0.244
        foot_clearance = 0.06

        for leg in legs:
            leg_theta = (leg.theta/(2*no_of_points))* 2*PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center + leg.y_shift

            leg.x, leg.y, leg.z = np.array([[np.cos(leg.phi),0,np.sin(leg.phi)],[0,1,0],[-np.sin(leg.phi),0, np.cos(leg.phi)]])@np.array([x,y,0])
            leg.z = leg.z + leg.z_shift

            leg.motor_knee, leg.motor_hip,leg.motor_abduction = self._inverse_3D(leg.x, leg.y, leg.z, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]


        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        
        return leg_abduction_angles,leg_motor_angles
    

    def _inverse_stoch2(self, x,y,Leg):

        l1 =    Leg[0]
        l2 =    Leg[1]
        l4 =    Leg[2]
        l5 =    Leg[3]
        le =    Leg[5]
        tq1 =   Leg[6]
        tq2 =   Leg[7]
        delta = Leg[4]
        xb = [[0,0],[0,0]]
        yb = [[0,0],[0,0]]
        phid = [0,0];psi = [0,0]; theta = [0,0]
        R_base = [[0,0],[0.035,0]]
        xb[0] = R_base[0][0];xb[1] = R_base[1][0]
        yb[0] = R_base[0][1];yb[1] = R_base[1][1]
        l3 = np.sqrt((x-xb[0])**2+(y-yb[0])**2)
        theta[0] = np.arctan2((y-yb[0]),(x-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = np.arccos(zeta)
        psi[0] = np.arctan2(l2*np.sin(phid[0]),(l1+l2*np.cos(phid[0])))
        q1 = theta[0] - psi[0]
        q2 = q1 + phid[0]
        xm = l1*np.cos(q1)+l2*np.cos(q2)
        ym = l1*np.sin(q1)+l2*np.sin(q2)
        xi = (xm+xb[0])
        yi = (ym+yb[0])

        xi = xb[0] + l1*np.cos(q1) + 0.04*np.cos(q2-tq1)
        yi = yb[0] + l1*np.sin(q1) + 0.04*np.sin(q2-tq1)
        R = [xi,yi]
        l6 = np.sqrt(((xi-xb[1])**2+(yi-yb[1])**2))
        theta[1] = np.arctan2((yi-yb[1]),(xi-xb[1]))
        Zeta = (l6**2 - l4**2 - l5**2)/(2*l5*l4)
        leg = 'left'
        Zeta = np.sign(Zeta) if abs(Zeta) > 1 else Zeta
        phid[1] = np.arccos(Zeta)
        psi[1] = np.arctan2(l5*np.sin(phid[1]),(l4+l5*np.cos(phid[1])))
        q3 = theta[1]+psi[1]
        q4 = q3-phid[1]
        xm = l4*np.cos(q3)+l5*np.cos(q4)+xb[1]
        ym = l4*np.sin(q3)+l5*np.sin(q4)+yb[1]

        if Zeta == 1:
            [q1, q2] = self._inverse_new(xm,ym,delta,Leg)

        return [q3, q1, q4, q2]

    def _inverse_new(self, xm,ym,delta,Leg):

        l1 = Leg[0]
        l2 = Leg[1]-Leg[4]
        l4 = Leg[2]
        l5 = Leg[3]
        delta = Leg[4]
        xb = [[0,0],[0,0]]
        yb = [[0,0],[0,0]]
        phid = [0,0];psi = [0,0]; theta = [0,0]
        R_base = [[1,0],[-1,0]]
        xb[0] = R_base[0][0];xb[1] = R_base[1][0]
        yb[0] = R_base[0][1];yb[1] = R_base[1][1]
        l3 = np.sqrt((xm-xb[0])**2+(ym-yb[0])**2)
        theta[0] = np.arctan2((ym-yb[0]),(xm-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = np.arccos(zeta)
        psi[0] = np.arctan2(l2*np.sin(phid[0]),(l1+l2*np.cos(phid[0])))
        q1 = theta[0] + psi[0]
        q2 = q1 - phid[0]
        xm = l1*np.cos(q1)+l2*np.cos(q2)
        ym = l1*np.sin(q1)+l2*np.sin(q2)

        return [q1,q2]

    def _inverse_3D(self, x, y, z, Leg):
        theta = np.arctan2(z,-y)
        new_coords = np.array([x,-y/np.cos(theta) - 0.035,z])
        motor_knee, motor_hip, _, _ = self._inverse_stoch2(new_coords[0], -new_coords[1], Leg)
        return [motor_knee, motor_hip, theta]



    def _update_leg_phi_val(self, leg_phi):
        
        self.front_right.phi =  leg_phi[0]
        self.front_left.phi = leg_phi[1]
        self.back_right.phi =   leg_phi[2]
        self.back_left.phi =  leg_phi[3]
    
    def _update_leg_phi(self, radius):
        if(radius >= 0):
            self.front_left.phi =  np.arctan2(self.body_length/2, radius + self.body_width/2)
            self.front_right.phi = -np.arctan2(self.body_length/2, radius - self.body_width/2)
            self.back_left.phi = -np.arctan2(self.body_length/2, radius + self.body_width/2)
            self.back_right.phi =  np.arctan2(self.body_length/2, radius - self.body_width/2)
            PHI = [self.front_right.phi,self.front_left.phi,self.back_right.phi,self.back_left.phi]
            np.save("Sim_States/action_PHI_"+str(radius)+".npy", PHI)

        if(radius<0):
            newr = -1*radius
            self.front_right.phi =  np.arctan2(self.body_length/2, newr + self.body_width/2)
            self.front_left.phi = -np.arctan2(self.body_length/2, newr - self.body_width/2)
            self.back_right.phi = -np.arctan2(self.body_length/2, newr + self.body_width/2)
            self.back_left.phi =  np.arctan2(self.body_length/2, newr - self.body_width/2)
            PHI = [self.front_right.phi, self.front_left.phi, self.back_right.phi, self.back_left.phi]
            np.save("Sim_States/action_PHI_" + str(radius) + ".npy", PHI)

    def _update_leg_step_length_val(self, step_length):
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]


    def _update_leg_step_length(self, step_length, radius):
        if(abs(radius) <= 0.12):
            self.front_right.step_length = step_length
            self.front_left.step_length = step_length 
            self.back_right.step_length = step_length 
            self.back_left.step_length = step_length

        elif(radius > 0.12):
            self.front_right.step_length = step_length * (radius - self.body_width/2)/radius
            self.front_left.step_length = step_length * (radius + self.body_width/2)/radius
            self.back_right.step_length = step_length * (radius - self.body_width/2)/radius
            self.back_left.step_length = step_length * (radius + self.body_width/2)/radius

            SL = [self.front_right.step_length, self.front_left.step_length,self.back_right.step_length,self.back_left.step_length]
            np.save("Sim_States/action_SL_"+str(radius)+".npy", SL)

        elif(radius < -0.12):
            newr = radius*-1
            self.front_left.step_length = step_length * (newr- self.body_width/2)/newr
            self.front_right.step_length = step_length * (newr + self.body_width/2)/newr
            self.back_left.step_length = step_length * (newr - self.body_width/2)/newr
            self.back_right.step_length = step_length *(newr + self.body_width/2)/newr

            SL = [self.front_right.step_length, self.front_left.step_length, self.back_right.step_length,
                  self.back_left.step_length]
            np.save("Sim_States/action_SL_" + str(radius) + ".npy", SL)

    def _update_leg_step_length_footstep(legs, footstep, last_footstep):
        legs.front_left.step_length = ((footstep.front_left.x - last_footstep.front_left.x)**2 + (footstep.front_left.z - last_footstep.front_left.z)**2)**0.5
        legs.front_right.step_length = ((footstep.front_right.x - last_footstep.front_right.x)**2 + (footstep.front_right.z - last_footstep.front_right.z)**2)**0.5
        legs.back_left.step_length = ((footstep.back_left.x - last_footstep.back_left.x)**2 + (footstep.back_left.z - last_footstep.back_left.z)**2)**0.5
        legs.back_right.step_length = ((footstep.back_right.x - last_footstep.back_right.x)**2 + (footstep.back_right.z - last_footstep.back_right.z)**2)**0.5

    def _update_leg_transformation_matrix(legs, footstep, last_footstep):
        
        pass
def constrain_abduction(angle):
    if(angle < 0):
        angle = 0
    elif(angle > 0.35):
        angle = 0.35
    return angle

if(__name__ == "__main__"):
    # action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828, -0.06466855, -0.45247894,  0.72117291, -0.11068088])
    walkcon = WalkingController(phase=[PI,0,0,PI])
    walkcon._update_leg_step_length(0.068*2, 0.4)
    walkcon._update_leg_phi(0.4)

