# Class definitions for the kinematics of the serial-2R manipulator
# and the modified five-bar manipulator (Stoch-2 leg)
#
#
# Created : 27 March, 2019
# Author: Aditya
# 

import numpy as np
import math


# Serial2R Kinematics class
# Functions include : Forward kinematics, inverse kinematics, Jacobian w.r.t the end-effector
# Assumes absolute angles between the links

class Serial2RKin():
    def __init__(self, 
            base_pivot=[0,0], 
            link_lengths=[0.3,0.3]):
        self.link_lengths = link_lengths
        self.base_pivot = base_pivot


    def inverseKinematics(self, ee_pos, branch=1):
        '''
        Inverse kinematics of a serial 2-R manipulator
        Inputs:
        -- base_pivot: Position of the base pivot (in Cartesian co-ordinates)
        -- link_len: Link lenghts [l1, l2]
        -- ee_pos: position of the end-effector [x, y] (Cartesian co-ordinates)

        Output:
        -- Solutions to both the branches of the IK. Angles specified in radians.
        -- Note that the angle of the knee joint is relative in nature.
        '''
        valid = True
        q = np.zeros(2, float)
        [x, y] = ee_pos - self.base_pivot
        [l1, l2] = self.link_lengths
        # Check if the end-effector point lies in the workspace of the manipulator
        if ((x**2 + y**2) > (l1+l2)**2) or ((x**2 + y**2) < (l1-l2)**2):
            #print("Point is outside the workspace")
            valid=False
            return valid, q
        a = 2*l2*x
        b = 2*l2*y
        c = l1**2 - l2**2 - x**2 - y**2
        if branch == 1:
            q1_temp = math.atan2(b, a) + math.acos(-c/math.sqrt(a**2 + b**2))
        elif branch == 2:
            q1_temp = math.atan2(b, a) - math.acos(-c/math.sqrt(a**2 + b**2))

        q[0] = math.atan2(y - l2*math.sin(q1_temp), x - l2*math.cos(q1_temp))
        q[1] = q1_temp - q[0]
        valid = True
        return valid, q
    

    def forwardKinematics(self, q):
        '''
        Forward Kinematics of the serial-2R manipulator
        Args:
        --- q : A vector of the joint angles [q_hip, q_knee], where q_knee is relative in nature
        Returns:
        --- x : The position vector of the end-effector
        '''
        [l1, l2] = self.link_lengths
        x = self.base_pivot + l1*np.array([math.cos(q[0]), math.sin(q[0])]) + l2*np.array([math.cos(q[0] + q[1]), math.sin(q[0] + q[1])])
        return x


    def Jacobian(self, q):
        '''
        Provides the Jacobian matrix for the end-effector
        Args:
        --- q : The joint angles of the manipulator [q_hip, q_knee], where the angle q_knee is specified relative to the thigh link
        Returns:
        --- mat : A 2x2 velocity Jacobian matrix of the manipulator
        '''
        [l1, l2] = self.link_lengths
        mat = np.zeros([2,2])
        mat[0,0]= -l1*math.sin(q[0]) - l2*math.sin(q[0] + q[1])
        mat[0,1] = - l2*math.sin(q[0] + q[1])
        mat[1,0] = l1*math.cos(q[0]) + l2*math.cos(q[0] + q[1])
        mat[1,1] = l2*math.cos(q[0] + q[1])
        return mat


class Stoch2Kinematics(object):
    '''
    Class to implement the position and velocity kinematics for the Stoch 2 leg
    Position kinematics: Forward kinematics, Inverse kinematics
    Velocity kinematics: Jacobian
    '''
    def __init__(self,
            base_pivot1=[0,0],
            base_pivot2=[0.035, 0],
            link_parameters=[0.12, 0.15015,0.04,0.11187, 0.15501, 0.04, 0.2532, 2.803]):
        self.base_pivot1 = base_pivot1
        self.base_pivot2 = base_pivot2
        self.link_parameters = link_parameters


    def inverseKinematics(self, x):
        '''
        Inverse kinematics of the Stoch 2 leg
        Args:
        -- x : Position of the end-effector
        Return:
        -- valid : Specifies if the result is valid
        -- q : The joint angles in the sequence [theta1, phi2, phi3, theta4], where the ith angle
               is the angle of the ith link measured from the horizontal reference. q will be zero
               when the inverse kinematics solution does not exist.
        '''
        valid = False
        q = np.zeros(4)
        [l1, l2, l2a, l2b, l3, l4, alpha1, alpha2] = self.link_parameters
        leg1 = Serial2RKin(self.base_pivot1, [l1,l2])
        leg2 = Serial2RKin(self.base_pivot2, [l4, l3])
        valid1, q1 = leg1.inverseKinematics(x, branch=1)
        if not valid1:
            return valid, q
        p1 = self.base_pivot1 \
             + l1*np.array([math.cos(q1[0]), math.sin(q1[0])]) \
             + l2a*np.array([math.cos(q1[0] + q1[1] - alpha1), math.sin(q1[0] + q1[1] - alpha1)])
        valid2, q2 = leg2.inverseKinematics(p1, branch=2)
        if not valid2:
            return valid, q
        valid = True
        # Convert all angles to absolute reference
        q = [q1[0], q1[0]+q1[1], q2[0]+q2[1], q2[0]]
        return valid, q


    def forwardKinematics(self, q):
        '''
        Forward kinematics of the Stoch 2 leg
        Args:
        -- q : Active joint angles, i.e., [theta1, theta4], angles of the links 1 and 4 (the driven links)
        Return:
        -- valid : Specifies if the result is valid
        -- x : End-effector position
        '''
        valid = False
        x = np.zeros(2)
        [l1, l2, l2a, l2b, l3, l4, alpha1, alpha2] = self.link_parameters
        p1 = self.base_pivot1 + l1*np.array([math.cos(q[0]), math.sin(q[0])])
        p2 = self.base_pivot2 + l4*np.array([math.cos(q[1]), math.sin(q[1])])
        leg = Serial2RKin(p1, [l2a, l3])
        valid, q = leg.inverseKinematics(p2, branch=1)
        if not valid:
            return valid, x
        x = p1 \
            + l2a*np.array([math.cos(q[0]), math.sin(q[0])]) \
            + l2b*np.array([math.cos(q[0] + math.pi - alpha2), math.sin(q[0] + math.pi - alpha2)])
        valid = True
        return valid, x


    def Jacobian(self, x):
        '''
        Provides the forward velocity Jacobian matrix given the end-effector position
        Inverse-kinematics is perfomed to obtain the joint angles
        Args:
        --- x: The position vector of the end-effector
        Returns:
        --- mat: A 2x2 Jacobian matrix
        '''
        mat = np.zeros([2,2])
        valid = False
        [l1, l2, l2a, l2b, l3, l4, alpha1, alpha2] = self.link_parameters
        valid_IK, q = self.inverseKinematics(x)
        if not valid_IK:
            return valid, mat
        
        [th1, phi2, phi3, th4] = q
        J_xth = np.array([[-l1*math.sin(th1), 0],\
                [l1*math.cos(th1), 0]])
        J_xphi = np.array([[0, -l2a*math.sin(phi2 - alpha1) -l2b*math.sin(phi2 - alpha1 + math.pi - alpha2)],\
                [0, l2a*math.cos(phi2 - alpha1) + l2b*math.cos(phi2 - alpha1 + math.pi - alpha2)]])
        K_th = np.array([[-l1*math.sin(th1), l4*math.sin(th4)],\
                [l1*math.cos(th1), -l4*math.cos(th4)]])
        K_phi = np.array([[-l2a*math.sin(phi2 - alpha1), l3*math.sin(phi3) ],\
                [l2a*math.cos(phi2 - alpha1), -l3*math.cos(phi3)]])

        K_phi_inv = np.linalg.inv(K_phi)

        mat = J_xth - J_xphi*(K_phi_inv*K_th)

        return mat



# End of file
