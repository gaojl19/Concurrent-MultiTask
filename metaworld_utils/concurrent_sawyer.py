from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import time


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

BAR = 0.025

class ConcurrentSawyerEnv(SawyerXYZEnv):
    def __init__(
            self,
            random_init=True,
            task_types=["push-1", "push-2"],
            obs_type='plain',
            goal_low=(-0.1, 0.6, 0.05),
            goal_high=(0.2, 0.9, 0.3),
            liftThresh = 0.04,
            sampleMode='equal',
            rewMode = 'orig',
            rotMode='fixed',#'fixed',
            **kwargs
    ):
        
        '''
            observation [   sawyer_x, sawyer_y, sawyer_z, 0:3
                            obj1_x, obj1_y, obj1_z, 3:6
                            obj2_x, obj2_y, obj2_z, 6:9
                            goal1_x, goal1_y, goal1_z, 9:12
                            goal2_x, goal2_y, goal2_z  12:15 ]
        '''
        self.quick_init(locals())

        hand_low=(-0.5, 0.40, 0.05)
        hand_high=(0.5, 1, 0.5)
        obj_low=(-0.1, 0.4, 0.02)
        obj_high=(0.1, 0.7, 0.02)
        self.fixed_point=(0.03662452, 0.67840914, 0.05201501) # the hand should return to this place

        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        self.hand_init_pos = np.array([0, .6, .2])
        
        # push-1
        self.init_config1 = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.02]),
        }
        
        # push-2
        self.init_config2 = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([-0.1, 0.5, 0.02]),
        }
        
    
        self.goal = np.array([0.05, 0.75, 0.02, 0.2, 0.55, 0.02])
            
        
        self.obj_init_angle1 = self.init_config1['obj_init_angle']
        self.obj_init_pos1 = self.init_config1['obj_init_pos']
        
        self.obj_init_angle2 = self.init_config2['obj_init_angle']
        self.obj_init_pos2 = self.init_config2['obj_init_pos']

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        if goal_low is None:
            goal_low = self.hand_low
        
        if goal_high is None:
            goal_high = self.hand_high

        # self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 1000
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.sampleMode = sampleMode
        self.task_types = task_types
        
        self.random_init = random_init
        
        # separate observations
        # TODO
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low,)),
                np.hstack((self.hand_high, obj_high,)),
            )
        elif self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            raise NotImplementedError('If you want to use an observation\
                with_obs_idx, please discretize the goal space after instantiate an environment.')
        self.num_resets = 0
        self.reset(seed=0)

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/concurrent_sawyer.xml')

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        # print(ob)
        obs_dict = self._get_obs_dict()
        reward, reachRew1, reachDist1, pushRew1, pushDist1, reachRew2, reachDist2, pushRew2, pushDist2 = self.compute_reward(action, obs_dict, mode=self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False

        # goal_dist = placingDist if self.task_type == 'pick_place' else pushDist
        # if self.task_type == 'reach':
        #     success = float(reachDist <= 0.05)
        # elif self.task_type == 'pick':
        #     success = self.pickCompleted
        # elif self.task_type == 'push':
        #     success = float(goal_dist <= 0.07)
        # else:
        #     success = float(goal_dist <= 0.07)
        success1 = float(pushDist1 <= 0.07)
        success2 = float(pushDist2 <= 0.07)
        if isinstance(obs_dict, dict):
            obs = obs_dict['state_observation']
        info = {'epRew' : reward, 'pushDist1': pushDist1, 'pushDist2': pushDist2, 'success_push_1': success1, 'success_push_2': success2, 'goal': self.goal, 'push1': obs[:3], 'push2': obs[3:6]}
        info['goal'] = self.goal
        return ob, reward, done, info
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos1 = self.data.get_geom_xpos('objGeom1')
        objPos2 = self.data.get_geom_xpos('objGeom2')
        flat_obs = np.concatenate((hand, objPos1, objPos2))
        if self.obs_type == 'with_goal_and_id':
            return np.concatenate([
                    flat_obs,
                    self._state_goal,
                    self._state_goal_idx
                ])
        elif self.obs_type == 'with_goal':
            return np.concatenate([
                    flat_obs,
                    self._state_goal
                ])
        elif self.obs_type == 'plain':
            return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
        else:
            return np.concatenate([flat_obs, self._state_goal_idx])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos1 =  self.data.get_geom_xpos('objGeom1')
        objPos2 =  self.data.get_geom_xpos('objGeom2')
        
        flat_obs = np.concatenate((hand, objPos1, objPos2))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=np.concatenate((objPos1, objPos2)),
        )

    def _get_info(self):
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal_push-1')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('goal_push-2')] = (
            goal[3:]
        )
        self.data.site_xpos[self.model.site_name2id('goal_fixed_point')] = (
            self.fixed_point
        )
        
    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos1 =  self.data.get_geom_xpos('objGeom1')
        self.data.site_xpos[self.model.site_name2id('objSite1')] = (
            objPos1
        )
        
        objPos2 =  self.data.get_geom_xpos('objGeom2')
        self.data.site_xpos[self.model.site_name2id('objSite2')] = (
            objPos2
        )


    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)


    def sample_goals(self, batch_size):
        # Required by HER-TD3
        goals = self.sample_goals_(batch_size)
        if self.discrete_goal_space is not None:
            goals = [self.discrete_goals[g].copy() for g in goals]
        return {
            'state_desired_goal': goals,
        }

    def sample_task(self):
        idx = self.sample_goals_(1)
        return self.discrete_goals[idx]

    def adjust_initObjPos(self, orig_init_pos, obj_name, obj_geom_name):
        #This is to account for meshes for the geom and object are not aligned
        #If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com(obj_name)[:2] - self.data.get_geom_xpos(obj_geom_name)[:2]
        adjustedPos = orig_init_pos[:2] + diff

        #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos(obj_geom_name)[-1]]
    
    def reset(self, seed, push_2_init=False):
        self.sim.reset()
        ob = self.reset_model(seed=seed, push_2_init=push_2_init)
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self, seed=0, push_2_init=False):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos1 = self.adjust_initObjPos(self.init_config1['obj_init_pos'], 'obj1', 'objGeom1')
        self.obj_init_pos2 = self.adjust_initObjPos(self.init_config2['obj_init_pos'], 'obj2', 'objGeom2')
        self.obj_init_angle1 = self.init_config1['obj_init_angle']
        self.obj_init_angle2 = self.init_config2['obj_init_angle']
        self.objHeight1 = self.data.get_geom_xpos('objGeom1')[2]
        self.objHeight2 = self.data.get_geom_xpos('objGeom2')[2]
        
        self.heightTarget1 = self.objHeight1 + self.liftThresh
        self.heightTarget2 = self.objHeight2 + self.liftThresh

        np.random.seed(seed)
        if self.random_init:
            epsilon = np.random.uniform(
                low=np.array([-BAR, -BAR, 0]),
                high=np.array([BAR, BAR, 0]),
                size=np.array([0, 0, 0]).size)
            self.init_config1["obj_init_pos"] += epsilon
            epsilon = np.random.uniform(
                low=np.array([-BAR, -BAR, 0]),
                high=np.array([BAR, BAR, 0]),
                size=np.array([0, 0, 0]).size)
            self.init_config2["obj_init_pos"] += epsilon
            epsilon = np.random.uniform(
                low=np.array([-BAR, -BAR, 0, -BAR, -BAR, 0]),
                high=np.array([BAR, BAR, 0, BAR, BAR, 0]),
                size=np.array([0, 0, 0, 0, 0, 0]).size)
            self.goal += epsilon
            
            
        # if self.random_init:
            # goal_pos = np.random.uniform(
            #     self.obj_and_goal_space.low,
            #     self.obj_and_goal_space.high,
            #     size=(self.obj_and_goal_space.low.size),
            # )
            # self._state_goal = goal_pos[3:]
            # while np.linalg.norm(goal_pos[:2] - self._state_goal[:2]) < 0.15:
            #     goal_pos = np.random.uniform(
            #         self.obj_and_goal_space.low,
            #         self.obj_and_goal_space.high,
            #         size=(self.obj_and_goal_space.low.size),
            #     )
            #     self._state_goal = goal_pos[3:]
            # if self.task_type == 'push':
            #     self._state_goal = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
            #     self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            # else:
            #     self._state_goal = goal_pos[-3:]
            #     self.obj_init_pos = goal_pos[:3]
        
        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos1)
        self._set_obj_xyz(self.obj_init_pos2)
        
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.curr_path_length = 0
        # self.maxReachDist = np.linalg.norm(self.init_fingerCOM - np.array(self._state_goal))
        # self.maxPickDist = np.linalg.norm(self.init_fingerCOM - np.array(self.obj_init_pos))
        self.maxPushDist1 = np.linalg.norm(self.obj_init_pos1[:2] - np.array(self._state_goal)[:2])
        self.maxPushDist2 = np.linalg.norm(self.obj_init_pos2[:2] - np.array(self._state_goal)[3:5])
        # self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        # self.target_rewards = [1000*self.maxPushDist + 1000*2]
        # if self.task_type == 'reach':
        #     idx = 1
        # elif self.task_type == 'push':
        #     idx = 2
        #     # move to fixed point
        #     # self.move_to_fixedpoint()   
        # else:
        #     idx = 0
        # self.target_reward = self.target_rewards[idx]
        # self.move_to_fixedpoint()
        
        if push_2_init:
            print("push_2 init!")
            # self.move_to_fixedpoint()
        self.num_resets += 1
        
        self.stablize() # stablizing the hand to make it easier
        return self._get_obs()

    def reset_model_to_idx(self, idx):
        raise NotImplementedError('This API is deprecated! Please explicitly\
            call `set_goal_` then reset the environment.')

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs, task_type=self.task_type)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode = 'general', task_type='reach'):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        handPos = obs[:3]
        objPos1 = obs[3:6]
        objPos2 = obs[6:9]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        # heightTarget = self.heightTarget
        goal = self._state_goal
        
        goal1 = goal[:3]
        goal2 = goal[3:]

        # def compute_reward_reach(actions, obs, mode):
        #     c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
        #     reachDist = np.linalg.norm(fingerCOM - goal)
        #     # reachRew = -reachDist
        #     # if reachDist < 0.1:
        #     #     reachNearRew = 1000*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
        #     # else:
        #     #     reachNearRew = 0.
        #     reachRew = c1*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
        #     reachRew = max(reachRew, 0)
        #     # reachNearRew = max(reachNearRew,0)
        #     # reachRew = -reachDist
        #     reward = reachRew# + reachNearRew
        #     return [reward, reachRew, reachDist, None, None, None, None, None, None]

        def compute_reward_push_1(actions, obs, mode):
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            assert np.all(goal1[:3] == self.get_site_pos('goal_push-1'))
            reachDist = np.linalg.norm(fingerCOM - objPos1)
            pushDist = np.linalg.norm(objPos1[:2] - goal1[:2])
            # print("push distance: ", pushDist, objPos1)
            
            # add return distance
            returnDist = np.linalg.norm(handPos - self.fixed_point)
    
            reachRew = -reachDist
            returnRew = -returnDist
            if reachDist < 0.05:
                # pushRew = -pushDist
                pushRew = 1000*(self.maxPushDist1 - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
                returnRew = 0
            reward = reachRew + pushRew
            return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None, None]
        
        def compute_reward_push_2(actions, obs, mode):
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            assert np.all(goal2[:3] == self.get_site_pos('goal_push-2'))
            reachDist = np.linalg.norm(fingerCOM - objPos2)
            pushDist = np.linalg.norm(objPos2[:2] - goal2[:2])
            
            # add return distance
            returnDist = np.linalg.norm(handPos - self.fixed_point)
            reachRew = -reachDist
            returnRew = -returnDist
            if reachDist < 0.05:
                # pushRew = -pushDist
                pushRew = 1000*(self.maxPushDist2 - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
                returnRew = 0
            reward = reachRew + pushRew + returnRew
            return [reward, None, None, None, None, reachRew, reachDist, pushRew, pushDist]
        
        # def compute_reward_pick_place(actions, obs, mode):
        #     reachDist = np.linalg.norm(objPos - fingerCOM)
        #     placingDist = np.linalg.norm(objPos - goal)

        #     assert np.all(goal == self.get_site_pos('goal_pick_place'))

        #     def reachReward():
        #         reachRew = -reachDist# + min(actions[-1], -1)/50
        #         reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
        #         zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
        #         if reachDistxy < 0.05: #0.02
        #             reachRew = -reachDist
        #         else:
        #             reachRew =  -reachDistxy - 2*zRew
        #         #incentive to close fingers when reachDist is small
        #         if reachDist < 0.05:
        #             reachRew = -reachDist + max(actions[-1],0)/50
        #         return reachRew , reachDist

        #     def pickCompletionCriteria():
        #         tolerance = 0.01
        #         if objPos[2] >= (heightTarget- tolerance):
        #             return True
        #         else:
        #             return False

        #     if pickCompletionCriteria():
        #         self.pickCompleted = True


        #     def objDropped():
        #         return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02) 
        #         # Object on the ground, far away from the goal, and from the gripper
        #         #Can tweak the margin limits
           
        #     def objGrasped(thresh = 0):
        #         sensorData = self.data.sensordata
        #         return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        #     def orig_pickReward():       
        #         # hScale = 50
        #         hScale = 100
        #         # hScale = 1000
        #         if self.pickCompleted and not(objDropped()):
        #             return hScale*heightTarget
        #         # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        #         elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        #             return hScale* min(heightTarget, objPos[2])
        #         else:
        #             return 0

        #     def general_pickReward():
        #         hScale = 50
        #         if self.pickCompleted and objGrasped():
        #             return hScale*heightTarget
        #         elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
        #             return hScale* min(heightTarget, objPos[2])
        #         else:
        #             return 0

        #     def placeReward():
        #         # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
        #         c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
        #         if mode == 'general':
        #             cond = self.pickCompleted and objGrasped()
        #         else:
        #             cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
        #         if cond:
        #             placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
        #             placeRew = max(placeRew,0)
        #             return [placeRew , placingDist]
        #         else:
        #             return [0 , placingDist]

        #     reachRew, reachDist = reachReward()
        #     if mode == 'general':
        #         pickRew = general_pickReward()
        #     else:
        #         pickRew = orig_pickReward()
        #     placeRew , placingDist = placeReward()
        #     assert ((placeRew >=0) and (pickRew>=0))
        #     reward = reachRew + pickRew + placeRew
        #     return [reward, reachRew, reachDist, None, None, pickRew, placeRew, placingDist, None]

        # def compute_reward_pick(actions, obs, mode):
        #     reachDist = np.linalg.norm(objPos - fingerCOM)

        #     assert np.all(goal == self.get_site_pos('goal_pick'))

        #     def reachReward():
        #         reachRew = -reachDist# + min(actions[-1], -1)/50
        #         reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
        #         zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
        #         if reachDistxy < 0.05: #0.02
        #             reachRew = -reachDist
        #         else:
        #             reachRew =  -reachDistxy - 2*zRew
        #         #incentive to close fingers when reachDist is small
        #         if reachDist < 0.05:
        #             reachRew = -reachDist + max(actions[-1],0)/50
        #         return reachRew , reachDist

        #     def pickCompletionCriteria():
        #         tolerance = 0.01
        #         if objPos[2] >= (heightTarget- tolerance):
        #             return True
        #         else:
        #             return False

        #     if pickCompletionCriteria():
        #         self.pickCompleted = True

        #     def objDropped():
        #         return (objPos[2] < (self.objHeight + 0.005)) and (reachDist > 0.02) 
        #         # Object on the ground, far away from the goal, and from the gripper
        #         #Can tweak the margin limits
           
        #     def objGrasped(thresh = 0):
        #         sensorData = self.data.sensordata
        #         return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        #     def orig_pickReward():       
        #         # hScale = 50
        #         hScale = 100
        #         # hScale = 1000
        #         if self.pickCompleted and not(objDropped()):
        #             return hScale*heightTarget
        #         # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        #         elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        #             return hScale* min(heightTarget, objPos[2])
        #         else:
        #             return 0

        #     def general_pickReward():
        #         hScale = 50
        #         if self.pickCompleted and objGrasped():
        #             return hScale*heightTarget
        #         elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
        #             return hScale* min(heightTarget, objPos[2])
        #         else:
        #             return 0

        #     reachRew, reachDist = reachReward()
        #     if mode == 'general':
        #         pickRew = general_pickReward()
        #     else:
        #         pickRew = orig_pickReward()
        #     assert (pickRew>=0)
        #     c_large = 100
        #     reward = c_large * (reachRew + pickRew)
        #     return [reward, reachRew, reachDist, None, None, pickRew, None, None, None]

        # if task_type == 'reach':
        #     return compute_reward_reach(actions, obs, mode)
        # elif task_type == 'push':
        #     return compute_reward_push(actions, obs, mode)
        # elif task_type == 'pick':
        #     return compute_reward_pick(actions, obs, mode)
        # else:
        #     return compute_reward_pick_place(actions, obs, mode)
        
        ret1 = compute_reward_push_1(actions, obs, mode)
        ret2 = compute_reward_push_2(actions, obs, mode)
        
        # reward, reachRew1, reachDist1, pushRew1, pushDist1, reachRew2, reachDist2, pushRew2, pushDist2
        result = [None, None, None, None, None, None, None, None, None]
        result[0:5] = ret1[0:5]
        result[5:] = ret2[5:]
        result[0] += ret2[0]
        
        return result

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def move_to_fixedpoint(self):
        import json
        with open("./metaworld_utils/reach_to_fixed_point_traj.json", 'r') as fin:
            acs = json.load(fin)["actions"]
        fin.close()
        obs = []
        
        for action in acs:
            obs.append(self._get_obs())
            action = np.array(action)
            if self.rotMode == 'euler':
                action_ = np.zeros(7)
                action_[:3] = action[:3]
                action_[3:] = euler2quat(action[3:6])
                self.set_xyz_action_rot(action_)
            elif self.rotMode == 'fixed':
                self.set_xyz_action(action[:3])
            elif self.rotMode == 'rotz':
                self.set_xyz_action_rotz(action[:4])
            else:
                self.set_xyz_action_rot(action[:7])
            self.do_simulation([action[-1], -action[-1]])
            # The marker seems to get reset every time you do a simulation
            self._set_goal_marker(self._state_goal)
            
        # import json
        # demonstration = {"observations": [i.tolist() for i in obs]}
        # demo_json = json.dumps(demonstration, sort_keys=False, indent=4)
        # f = open("./log/simulator_fixed_point_traj.json", 'w')
        # f.write(demo_json)
    
    
    def stablize(self):
        for _ in range(10):
            action = np.array([0, 0, 0, 0])
            if self.rotMode == 'euler':
                action_ = np.zeros(7)
                action_[:3] = action[:3]
                action_[3:] = euler2quat(action[3:6])
                self.set_xyz_action_rot(action_)
            elif self.rotMode == 'fixed':
                self.set_xyz_action(action[:3])
            elif self.rotMode == 'rotz':
                self.set_xyz_action_rotz(action[:4])
            else:
                self.set_xyz_action_rot(action[:7])
            self.do_simulation([action[-1], -action[-1]])
            # The marker seems to get reset every time you do a simulation
            self._set_goal_marker(self._state_goal)
    
    def move_to_fixedpoint_2(self):
        import json
        with open("./metaworld_utils/reach_to_fixed_point_traj_2.json", 'r') as fin:
            acs = json.load(fin)["actions"]
        fin.close()
        obs = []
        
        for action in acs:
            obs.append(self._get_obs())
            action = np.array(action)
            if self.rotMode == 'euler':
                action_ = np.zeros(7)
                action_[:3] = action[:3]
                action_[3:] = euler2quat(action[3:6])
                self.set_xyz_action_rot(action_)
            elif self.rotMode == 'fixed':
                self.set_xyz_action(action[:3])
            elif self.rotMode == 'rotz':
                self.set_xyz_action_rotz(action[:4])
            else:
                self.set_xyz_action_rot(action[:7])
            self.do_simulation([action[-1], -action[-1]])
            # The marker seems to get reset every time you do a simulation
            self._set_goal_marker(self._state_goal)

    def log_diagnostics(self, paths = None, logger = None):
        pass
