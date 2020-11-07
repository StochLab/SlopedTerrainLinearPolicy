## Robust Quadrupedal Locomotion on Sloped Terrains: A Linear Policy Approach

### Introduction:

<p align="center">
   <img width="200" height="180" src="https://github.com/CPS-Quadruped/Stoch2_gym_env/blob/master/media/stoch2uphill.gif">
   <img width="200" height="180" src="https://github.com/CPS-Quadruped/Stoch2_gym_env/blob/master/media/stoch2arbitary.gif">
   <img width="200" height="180" src="https://github.com/CPS-Quadruped/Stoch2_gym_env/blob/master/media/stoch2stair1.gif">
   <img width="200" height="180" src="https://github.com/CPS-Quadruped/Stoch2_gym_env/blob/master/media/stoch2dist.gif">
</p>

The code base accompanying the paper with the above title, accepted in CoRL 2020 and to be published in PMLR. The preprint of the submitted work is available [here](https://arxiv.org/abs/2010.16342) and the supplementary video can be viewed [here](https://youtu.be/KdQn1e3rI7o).

### Getting Started:
To install the package and its dependenclies run to following command, inside the folder, Stoch2_gym_env:        
                
        python -m pip install .

The code base was tested with gym (0.17.2), pybullet (2.8.2) with a python version of 3.6.9. However it is expected to work fine for any future versions of these packages, though they havent been tested.



### Robots Tested for:

The following tables represent the evalualtion of our controller on different inclines across multiple orientations.The limiations (depicted by :x:) are due to factors like the kinematics limits, robot dimensions(height to width ratio), and dynamics of the robot.

### Stoch2:

|Orientation\Elevation| -13°| -11° |-9° |7° |-5° |5° |7° |9° |11° | 13°|
|:-------------:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:---:|:---:|:---:|
|0°|:heavy_check_mark:| :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:x: |
|30°|:heavy_check_mark:| :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:x: |
|60°|:heavy_check_mark:| :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:x: |
|90°|:x:|:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:x: |

### HyQ:

|Orientation\Elevation|-15° |-13° |-11° |-9°|9° |11° |13° |15°|
|:-------------:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|
|0°| :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |
|30°| :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |
|45°| :x: | :x: | :heavy_check_mark:|:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |


### Laikago:
|Orientation\Elevation|-15° |-13° |-11° |-9°|9° |11° |13° |15°|
|:-------------:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|
|0°| :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |
|30°| :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |
|45°| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:|:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:x: |




### To make an initial policy:
As explained in the paper, we take a guided learning approach wherein the role of a initial policy is quite crucial.To train your own initial policy,run the following command
        
        python create_initial_policy.py --policyName filename --robotName Stoch2

This saves the initial policy as *filename.npy* in the initial_policies folder.This file is to be loaded later as the initial polcy when you want train your own polciy.However there are a few initial policies present in the same folder which could be directly used to start the ARS training. 

| Parameter     |About        |  type |
|:-------------:|:-------------:| -----:|
|--policyName   | name of the intial policy | str |
|--robotName    | name of the robot (Stoch2/Laikago/HyQ) | str |

**Note:** The initial policies are by default saved in the *initial_policies* folder.

### To train the linear policy with ARS
This is where the ARS trainining starts,with the initial polciy trainied in the previous step.
        
        python trainStoch2_policy.py 

The above command starts the training for Stoch2 in the default settings and by far the best observed hyperparameters. However, the following parameters can also be customized in the training as desired by the user.The policies for Laikago and HyQ could be trained similarly by running the scripts *trainLaikago_policy.py* and *trainHyQ_policy.py* respectively.

| Parameter     |About        |  type |
|:-------------:|:-------------:| -----:|
|--render      | flag for rendering | bool |
|--policy      | initial polciy to start the training with|str|
| --logdir | Directory root to log policy files (npy)     |str |
| --lr | learning rate     |float |
| --noise | amount of random noise to be added to the weights|float |
| --msg |any message acompanying the training|str |
| --curi_learn |Number of learning iteration before changing the curriculum      |int |
| --eval_step |Number of policy iterations before a policy update     |int |
| --episode_length|Horizon of a episode|int |
| --domain_Rand|randomizatize the dynamics of the environment while training|int(ony 0 or 1)|
| --episode_length|Horizon of a episode|int |


For example,

      python trainStoch2_policy.py --lr 0.05 --noise 0.04 --logdir testDir --policy init_policy_TS.npy --msg "Training with some paramters" --episode_length 400

**Note:** 

1. The initial policies are by default loaded from the *initial_policies* folder and the log directory is saved inside the *experiments* folder.
2. Domain randomization has not yet been tested with the robots *Laikago and HyQ*.
3. The are a few other insignificant parameters which need not be changed for the training, for more info about the parameters run

        python trainStoch2_policy.py --help

## To conduct tests on a policy
To run a policy in default conditions, the following command is to be used.

        python testStoch2_policy.py

The policies for Laikago and HyQ could be tested similarly by running the scripts *testLaikago_policy.py* and *testHyQ_policy.py* respectively.The following test parameters can be changed while testing the policy,

| Parameter     |About        |  type | Allowed values|unit|
|:-------------:|:-------------:|:-----:|:---------:|:-----:|
|--PolicyDir | directory of the policy to be tested (best policies are loaded by default)| str |(check the experiments folder)| - |
|--Stairs | load staircase | bool |True or False|unitless|
|--WedgeIncline | the elevation angle of wedge | int |0,5,7,9,11,13,15|Degrees(°)|
|--WedgeOrientation| the yaw angle of wedge about world z axis | float | -90.0 to 90.0 |Degrees(°)|
|--EpisodeLength |number of gait steps of a episode| int |0 to inf|number of steps|
|--MotorStrength|maximum motor strength that could be applied| float |5.0 to 8.0|NewtonMetre(Nm)|
|--FrictionCoeff|coefficient of friction to be set| float |0.55 to 0.80|unitless|
|--FrontMass|mass to be loaded to the front half of the body| float |0.0 to 0.15|Kilograms(Kg)|
|--BackMass|mass to be loaded to the  rear half of the body| float |0.0 to 0.15|Kilograms(Kg)|
|--RandomTest|flag to activate random sampling| bool |True or False|unitless|
|--seed|seed for random sampling| int | - |unitless|
|--PerturbForce|perturbation force to applied perpendicular to the heading direction of the robot|float|-120 to 120|Newton(N)|
|--AddImuNoise| flag to add noise in IMU readings | bool |True or False|unitless|

Thus, for a 

1. custom test

        python testStoch2_policy.py --PolicyDir 23July3 --WedgeIncline 11 --WedgeOrientation 15 --FrontMass 0.1 --FrictionCoeff 0.6

2. random test

        python testStoch2_policy.py --PolicyDir 23July3 --RandomTest True --seed 100

## To conduct tests on a staircase
To run a policy on a staircase of fixed dimensions, the following command is to be used. As of now only avaialable for Stoch2.

        python testStoch2_policy.py --Stairs True
        
## To conduct tests on a arbitary slopes
To run a policy on a arbitary slope track, the following command is to be used. As of now only avaialable for Stoch2.

        python arbitary_slope_test.py

**Note:** 

1. The test policies are by default loaded from the path *experiments/**given_logdir_name**/iterations/best_policy.npy"*, if not specified it loads the best ever policy pre-trained by us.
2. In our method we only train for +ve roll and -ve pitch conditions of support plane, the trained policy is able to generalize for other conditions too.
3. Our env is not fully supported for training in downhill case, but you can evalute policy in downhill conditions.
4. The  features like stairs, domain parameters that could be randomized, and arbitary slopes have not yet been added in *testLaikago_policy.py and testHyQ_policy.py*.



