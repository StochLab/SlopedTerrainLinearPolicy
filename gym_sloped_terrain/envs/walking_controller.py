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
from utils.ik_class import Stoch2Kinematics
from utils.ik_class import LaikagoKinematics
from utils.ik_class import HyqKinematics
import numpy as np

PI = np.pi
no_of_points = 100


@dataclass
class leg_data:
    name: str
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    motor_abduction: float = 0.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    b: float = 1.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0


@dataclass
class robot_data:
    front_right: leg_data = leg_data('fr')
    front_left: leg_data = leg_data('fl')
    back_right: leg_data = leg_data('br')
    back_left: leg_data = leg_data('bl')


class WalkingController():
    def __init__(self,
                 gait_type='trot',
                 phase=[0, 0, 0, 0],
                 ):
        self._phase = robot_data(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = leg_data('fl')
        self.front_right = leg_data('fr')
        self.back_left = leg_data('bl')
        self.back_right = leg_data('br')
        self.gait_type = gait_type

        self.MOTOROFFSETS_Stoch2 = [2.3562, 1.2217]
        self.MOTOROFFSETS_Laikago = [0.87, 0.7]  # [np.pi*0.9, 0]#
        self.MOTOROFFSETS_HYQ = [1.57, 0]


        self.leg_name_to_sol_branch_HyQ = {'fl': 0, 'fr': 0, 'bl': 1, 'br': 1}
        self.leg_name_to_dir_Laikago = {'fl': 1, 'fr': -1, 'bl': 1, 'br': -1}
        self.leg_name_to_sol_branch_Laikago = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}

        self.body_width = 0.24
        self.body_length = 0.37
        self.Stoch2_Kin = Stoch2Kinematics()
        self.Laikago_Kin = LaikagoKinematics()
        self.Hyq_Kin = HyqKinematics()

    def update_leg_theta(self, theta):
        """ Depending on the gait, the theta for every leg is calculated"""

        def constrain_theta(theta):
            theta = np.fmod(theta, 2 * no_of_points)
            if (theta < 0):
                theta = theta + 2 * no_of_points
            return theta

        self.front_right.theta = constrain_theta(theta + self._phase.front_right)
        self.front_left.theta = constrain_theta(theta + self._phase.front_left)
        self.back_right.theta = constrain_theta(theta + self._phase.back_right)
        self.back_left.theta = constrain_theta(theta + self._phase.back_left)

    def initialize_elipse_shift(self, Yshift, Xshift, Zshift):
        '''
        Initialize desired X, Y, Z offsets of elliptical trajectory for each leg
        '''
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
        '''
        Initialize all the parameters of the leg trajectories
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            legs   : namedtuple('legs', 'front_right front_left back_right back_left')
        '''
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

    def run_elliptical_Traj_Stoch2(self, theta, action):
        '''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        '''
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.244
        foot_clearance = 0.06

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center + leg.y_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])
            leg.z = leg.z + leg.z_shift

            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Stoch2_Kin.inverseKinematics(leg.x, leg.y, leg.z)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Stoch2[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Stoch2[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip,
                            legs.back_right.motor_knee,
                            legs.front_left.motor_abduction, legs.front_right.motor_abduction,
                            legs.back_left.motor_abduction, legs.back_right.motor_abduction]

        return leg_motor_angles

    def run_elliptical_Traj_HyQ(self, theta, action):
        '''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        '''
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.7
        foot_clearance = 0.12

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) - leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center - leg.y_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], 
                
                [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])
            leg.z = leg.z - leg.z_shift
            leg.z = -1 * leg.z

            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Hyq_Kin.inverseKinematics(leg.x, leg.y, leg.z,
                                                                                                self.leg_name_to_sol_branch_HyQ[
                                                                                                    leg.name])
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_HYQ[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_HYQ[1]
            leg.motor_abduction = -1 * leg.motor_abduction

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip,
                            legs.front_right.motor_knee, legs.back_left.motor_hip, legs.back_left.motor_knee,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.front_left.motor_abduction,
                            legs.front_right.motor_abduction, legs.back_left.motor_abduction,
                            legs.back_right.motor_abduction]

        return leg_motor_angles

    def run_elliptical_Traj_Laikago(self, theta, action):
        '''
        Semi-elliptical trajectory controller
        Args:
            theta  : trajectory cycle parameter theta
            action : trajectory modulation parameters predicted by the policy
        Ret:
            leg_motor_angles : list of motors positions for the desired action [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        '''
        legs = self.initialize_leg_state(theta, action)

        y_center = -0.35
        foot_clearance = 0.1

        for leg in legs:
            leg_theta = (leg.theta / (2 * no_of_points)) * 2 * PI
            leg.r = leg.step_length / 2

            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) + leg.x_shift
                if leg_theta > PI:
                    flag = 0
                else:
                    flag = 1
                y = foot_clearance * np.sin(leg_theta) * flag + y_center + leg.y_shift

            leg.x, leg.y, leg.z = np.array(
                [[np.cos(leg.phi), 0, np.sin(leg.phi)], [0, 1, 0], [-np.sin(leg.phi), 0, np.cos(leg.phi)]]) @ np.array(
                [x, y, 0])

            leg.z = leg.z + leg.z_shift

            if leg.name == "fl" or leg.name == "bl":
                leg.z = -leg.z
        
            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self.Laikago_Kin.inverseKinematics(leg.x, leg.y, leg.z,
                                                                                                    self.leg_name_to_sol_branch_Laikago[
                                                                                                        leg.name])

            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS_Laikago[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS_Laikago[1]
            leg.motor_abduction = leg.motor_abduction * self.leg_name_to_dir_Laikago[leg.name]
            leg.motor_abduction = leg.motor_abduction + 0.07


        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_left.motor_abduction,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.back_right.motor_abduction,
                            legs.front_right.motor_hip, legs.front_right.motor_knee, legs.front_right.motor_abduction,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_left.motor_abduction]

        return leg_motor_angles

    def _update_leg_phi_val(self, leg_phi):
        '''
        Args:
             leg_phi : steering angles for each leg trajectories
        '''
        self.front_right.phi = leg_phi[0]
        self.front_left.phi = leg_phi[1]
        self.back_right.phi = leg_phi[2]
        self.back_left.phi = leg_phi[3]

    def _update_leg_step_length_val(self, step_length):
        '''
        Args:
            step_length : step length of each leg trajectories
        '''
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]


def constrain_abduction(angle):
    '''
    constrain abduction command with respect to the kinematic limits of the abduction joint
    '''
    if (angle < 0):
        angle = 0
    elif (angle > 0.35):
        angle = 0.35
    return angle


if (__name__ == "__main__"):
    walkcon = WalkingController(phase=[PI, 0, 0, PI])
    walkcon._update_leg_step_length(0.068 * 2, 0.4)
    walkcon._update_leg_phi(0.4)

