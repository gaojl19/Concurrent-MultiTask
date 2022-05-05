import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsertEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg import SawyerNutAssemblyEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_open import SawyerWindowOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer import SawyerHammerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_close import SawyerWindowCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn import SawyerDialTurnEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open import SawyerDrawerOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown import SawyerButtonPressTopdownEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking import SawyerBinPickingEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_push import SawyerStickPushEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull import SawyerStickPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press import SawyerButtonPressEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place import SawyerShelfPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoalEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_button import SawyerCoffeeButtonEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_push import SawyerCoffeePushEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_pull import SawyerCoffeePullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_open import SawyerFaucetOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_close import SawyerFaucetCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_unplug_side import SawyerPegUnplugSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_soccer import SawyerSoccerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_basketball import SawyerBasketballEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall import SawyerReachPushPickPlaceWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_back import SawyerPushBackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_out_of_hole import SawyerPickOutOfHoleEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_remove import SawyerShelfRemoveEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_disassemble_peg import SawyerNutDisassembleEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_lock import SawyerDoorLockEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_unlock import SawyerDoorUnlockEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_tool import SawyerSweepToolEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_wall import SawyerButtonPressWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_wall import SawyerButtonPressTopdownWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press import SawyerHandlePressEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull import SawyerHandlePullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_side import SawyerHandlePressSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_side import SawyerHandlePullSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide import SawyerPlateSlideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back import SawyerPlateSlideBackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_side import SawyerPlateSlideSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_side import SawyerPlateSlideBackSideEnv


SEPARATE_CONCURRENT = {
    "obs_dim": 15,
    "acs_dim": 4
}

CONCURRENT = {
    "obs_dim": 15,
    "acs_dim": 4,
}

def _hard_mode_args_kwargs(env_cls, key):
    kwargs = dict(random_init=True, obs_type='plain')
    if key == 'reach-v1' or key == 'reach-wall-v1':
        kwargs['task_type'] = 'reach'
    elif key == 'push-v1' or key == 'push-wall-v1':
        kwargs['task_type'] = 'push'
    elif key == 'pick-place-v1' or key == 'pick-place-wall-v1':
        kwargs['task_type'] = 'pick_place'
    return dict(args=[], kwargs=kwargs)


# customize our MT10 diverse 
DIVERSE_MT10_CLS_DICT= {
    'faucet-open-v1': SawyerFaucetOpenEnv,
    'box-close-v1': SawyerBoxCloseEnv,
    'window-close-v1': SawyerWindowCloseEnv,
    'soccer-v1': SawyerSoccerEnv,
    'dial-turn-v1': SawyerDialTurnEnv,
    'shelf-place-v1': SawyerShelfPlaceEnv,
    'handle-press-v1': SawyerHandlePressEnv,
    'plate-slide-v1': SawyerPlateSlideEnv,
    'drawer-open-v1': SawyerDrawerOpenEnv,
    'door-close-v1': SawyerDoorCloseEnv
}

DIVERSE_MT10_ARGS_KWARGS = {}
for key, env_cls in DIVERSE_MT10_CLS_DICT.items():
    DIVERSE_MT10_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)



'''
    customize MT10 similar task group
'''
SIMILAR_MT10_CLS_DICT = {
    'push-v1': SawyerReachPushPickPlaceEnv,
    'button-press-v1': SawyerButtonPressEnv,
    'coffee-push-v1': SawyerCoffeePushEnv,
    'soccer-v1': SawyerSoccerEnv,
    'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'sweep-v1': SawyerSweepEnv,
    'sweep-into-v1': SawyerSweepIntoGoalEnv,
    'plate-slide-v1': SawyerPlateSlideEnv,
    'plate-slide-side-v1': SawyerPlateSlideSideEnv,
    'drawer-close-v1': SawyerDrawerCloseEnv
}

SIMILAR_MT10_ARGS_KWARGS = {}
for key, env_cls in SIMILAR_MT10_CLS_DICT.items():
    SIMILAR_MT10_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)




'''
    Customize Similar but partially unsuccessful MT10 tasks
'''
FAIL_MT10_CLS_DICT = {
    'dissassemble-v1': SawyerNutDisassembleEnv,
    'assembly-v1': SawyerNutAssemblyEnv,
    'hand-insert-v1': SawyerHandInsertEnv,
    'pick-place-v1': SawyerReachPushPickPlaceEnv,
    'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'stick-push-v1': SawyerStickPushEnv,
    'pick-out-of-hole-v1': SawyerPickOutOfHoleEnv,
    'box-close-v1': SawyerBoxCloseEnv,
    'bin-picking-v1': SawyerBinPickingEnv,
    'basket-ball-v1': SawyerBasketballEnv,
}

FAIL_MT10_ARGS_KWARGS = {}
for key, env_cls in FAIL_MT10_CLS_DICT.items():
    FAIL_MT10_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)
    
    



'''
    Customize EASY_MT10 tasks: 9 successful : 1 unsuccessful
    Same as MT10 Similar
'''
EASY_MT10_CLS_DICT = {
    'push-v1': SawyerReachPushPickPlaceEnv,
    'button-press-v1': SawyerButtonPressEnv,
    'coffee-push-v1': SawyerCoffeePushEnv,
    'soccer-v1': SawyerSoccerEnv,
    'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'sweep-v1': SawyerSweepEnv,
    'sweep-into-v1': SawyerSweepIntoGoalEnv,
    'plate-slide-v1': SawyerPlateSlideEnv,
    'plate-slide-side-v1': SawyerPlateSlideSideEnv,
    'drawer-close-v1': SawyerDrawerCloseEnv
}

EASY_MT10_ARGS_KWARGS = {}
for key, env_cls in EASY_MT10_CLS_DICT.items():
    EASY_MT10_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)


'''
    Customize MEDIUM_MT10 tasks: 7 successful : 3 unsuccessful (which is average)
'''
MEDIUM_MT10_CLS_DICT = {
    'push-v1': SawyerReachPushPickPlaceEnv,
    'pick-place-v1': SawyerReachPushPickPlaceEnv,
    'drawer-open-v1': SawyerDrawerOpenEnv,
    'drawer-close-v1': SawyerDrawerCloseEnv,
    'basket-ball-v1': SawyerBasketballEnv,
    'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'coffee-push-v1': SawyerCoffeePushEnv,
    'coffee-pull-v1': SawyerCoffeePullEnv,
    'soccer-v1': SawyerSoccerEnv,
}

MEDIUM_MT10_ARGS_KWARGS = {}
for key, env_cls in MEDIUM_MT10_CLS_DICT.items():
    MEDIUM_MT10_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)



'''
    Customize HARD_MT10 tasks: 5 successful : 5 unsuccessful
'''
# HARD_MT10_CLS_DICT = {
#     'push-v1': SawyerReachPushPickPlaceEnv,
#     'pick-place-v1': SawyerReachPushPickPlaceEnv,
#     'drawer-open-v1': SawyerDrawerOpenEnv,
#     'button-press-wall-v1': SawyerButtonPressWallEnv,
#     'peg-insert-side-v1': SawyerPegInsertionSideEnv,
#     'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
#     'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
#     'coffee-pull-v1': SawyerCoffeePullEnv,
#     'button-press-v1': SawyerButtonPressEnv,
#     'stick-push-v1': SawyerStickPushEnv
# }

HARD_MT10_CLS_DICT = {
    'button-press-v1': SawyerButtonPressEnv,
    'button-press-wall-v1': SawyerButtonPressWallEnv,
    'push-v1': SawyerReachPushPickPlaceEnv,
    'drawer-open-v1': SawyerDrawerOpenEnv,
    'stick-push-v1': SawyerStickPushEnv,
    'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'pick-place-v1': SawyerReachPushPickPlaceEnv,
    'coffee-pull-v1': SawyerCoffeePullEnv,
    'peg-insert-side-v1': SawyerPegInsertionSideEnv,
}

HARD_MT10_ARGS_KWARGS = {}
for key, env_cls in HARD_MT10_CLS_DICT.items():
    HARD_MT10_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)
    
    

'''
    Customize MT40 tasks:   remove 7 unsuccessful agent(peg-unplug-side, disassemble, pick-out-of-hole, assembly, push-back, lever-pull, bin-picking), 
                            and 3 bad expert(hammer, stick-pull, peg-insert-side)
'''
MT40_CLS_DICT = {
    'reach-v1': SawyerReachPushPickPlaceEnv,
    'push-v1': SawyerReachPushPickPlaceEnv,
    'pick-place-v1': SawyerReachPushPickPlaceEnv,
    'reach-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
    'door-open-v1': SawyerDoorEnv,
    'door-close-v1': SawyerDoorCloseEnv,
    'drawer-open-v1': SawyerDrawerOpenEnv,
    'drawer-close-v1': SawyerDrawerCloseEnv,
    'button-press-topdown-v1': SawyerButtonPressTopdownEnv,
    'button-press-v1': SawyerButtonPressEnv,
    'button-press-topdown-wall-v1': SawyerButtonPressTopdownWallEnv,
    'button-press-wall-v1': SawyerButtonPressWallEnv,
    'window-open-v1': SawyerWindowOpenEnv,
    'window-close-v1': SawyerWindowCloseEnv,
    'plate-slide-v1': SawyerPlateSlideEnv,
    'plate-slide-side-v1': SawyerPlateSlideSideEnv,
    'plate-slide-back-v1': SawyerPlateSlideBackEnv, 
    'plate-slide-back-side-v1': SawyerPlateSlideBackSideEnv,
    'handle-press-v1': SawyerHandlePressEnv,
    'handle-pull-v1': SawyerHandlePullEnv,
    'handle-press-side-v1': SawyerHandlePressSideEnv,
    'handle-pull-side-v1': SawyerHandlePullSideEnv,
    'stick-push-v1': SawyerStickPushEnv,
    'basket-ball-v1': SawyerBasketballEnv,
    'soccer-v1': SawyerSoccerEnv,
    'faucet-open-v1': SawyerFaucetOpenEnv,
    'faucet-close-v1': SawyerFaucetCloseEnv,
    'coffee-push-v1': SawyerCoffeePushEnv,
    'coffee-pull-v1': SawyerCoffeePullEnv,
    'coffee-button-v1': SawyerCoffeeButtonEnv,
    'sweep-v1': SawyerSweepEnv,
    'sweep-into-v1': SawyerSweepIntoGoalEnv,
    'shelf-place-v1': SawyerShelfPlaceEnv,
    'dial-turn-v1': SawyerDialTurnEnv,
    'box-close-v1': SawyerBoxCloseEnv,
    'hand-insert-v1': SawyerHandInsertEnv,
    'door-lock-v1': SawyerDoorLockEnv,
    'door-unlock-v1': SawyerDoorUnlockEnv
}

MT40_ARGS_KWARGS = {}
for key, env_cls in MT40_CLS_DICT.items():
    MT40_ARGS_KWARGS[key] = _hard_mode_args_kwargs(env_cls, key)
    

ENV_TASK_DICT = {
    "mt10_diverse": DIVERSE_MT10_CLS_DICT,
    "mt10_similar": SIMILAR_MT10_CLS_DICT,
    "mt10_fail": FAIL_MT10_CLS_DICT,
    "mt10_medium": MEDIUM_MT10_CLS_DICT,
    "mt10_hard": HARD_MT10_CLS_DICT,
    "mt40": MT40_CLS_DICT
}

# dict(
#     train={
#         'reach-v1': SawyerReachPushPickPlaceEnv,
#         'push-v1': SawyerReachPushPickPlaceEnv,
#         'pick-place-v1': SawyerReachPushPickPlaceEnv,
#         'reach-wall-v1': SawyerReachPushPickPlaceWallEnv,
#         'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
#         'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
#         'door-open-v1': SawyerDoorEnv,
#         'door-close-v1': SawyerDoorCloseEnv,
#         'drawer-open-v1': SawyerDrawerOpenEnv,
#         'drawer-close-v1': SawyerDrawerCloseEnv,
#         'button-press_topdown-v1': SawyerButtonPressTopdownEnv,
#         'button-press-v1': SawyerButtonPressEnv,
#         'button-press-topdown-wall-v1': SawyerButtonPressTopdownWallEnv,
#         'button-press-wall-v1': SawyerButtonPressWallEnv,
#         'peg-insert-side-v1': SawyerPegInsertionSideEnv,
#         'peg-unplug-side-v1': SawyerPegUnplugSideEnv,
#         'window-open-v1': SawyerWindowOpenEnv,
#         'window-close-v1': SawyerWindowCloseEnv,
#         'dissassemble-v1': SawyerNutDisassembleEnv,
#         'hammer-v1': SawyerHammerEnv,
#         'plate-slide-v1': SawyerPlateSlideEnv,
#         'plate-slide-side-v1': SawyerPlateSlideSideEnv,
#         'plate-slide-back-v1': SawyerPlateSlideBackEnv, 
#         'plate-slide-back-side-v1': SawyerPlateSlideBackSideEnv,
#         'handle-press-v1': SawyerHandlePressEnv,
#         'handle-pull-v1': SawyerHandlePullEnv,
#         'handle-press-side-v1': SawyerHandlePressSideEnv,
#         'handle-pull-side-v1': SawyerHandlePullSideEnv,
#         'stick-push-v1': SawyerStickPushEnv,
#         'stick-pull-v1': SawyerStickPullEnv,
#         'basket-ball-v1': SawyerBasketballEnv,
#         'soccer-v1': SawyerSoccerEnv,
#         'faucet-open-v1': SawyerFaucetOpenEnv,
#         'faucet-close-v1': SawyerFaucetCloseEnv,
#         'coffee-push-v1': SawyerCoffeePushEnv,
#         'coffee-pull-v1': SawyerCoffeePullEnv,
#         'coffee-button-v1': SawyerCoffeeButtonEnv,
#         'sweep-v1': SawyerSweepEnv,
#         'sweep-into-v1': SawyerSweepIntoGoalEnv,
#         'pick-out-of-hole-v1': SawyerPickOutOfHoleEnv,
#         'assembly-v1': SawyerNutAssemblyEnv,
#         'shelf-place-v1': SawyerShelfPlaceEnv,
#         'push-back-v1': SawyerPushBackEnv,
#         'lever-pull-v1': SawyerLeverPullEnv,
#         'dial-turn-v1': SawyerDialTurnEnv,},
#     test={
#         'bin-picking-v1': SawyerBinPickingEnv,
#         'box-close-v1': SawyerBoxCloseEnv,
#         'hand-insert-v1': SawyerHandInsertEnv,
#         'door-lock-v1': SawyerDoorLockEnv,
#         'door-unlock-v1': SawyerDoorUnlockEnv,},
# )



