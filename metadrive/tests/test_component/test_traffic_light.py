from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.traffic_sign.base_stop_sign import BaseStopSign



def testBaseStopSign(render=False, debug=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "manual_control": True,
            "use_render": render,
            "debug": debug,
            "map": "X",
            "window_size": (1200, 800),
            "show_coordinates": True,
            "vehicle_config": {
                "show_lidar": True,
                "enable_reverse": True,
                "show_dest_mark": True
            },
        }
    )
    env.reset()
    try:
        # green
        env.reset()
        stop_sign = env.engine.spawn_object(BaseStopSign, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        for s in range(1, 1000):
            env.step([0, 1])
            # if env.vehicle.red_light or env.vehicle.yellow_light:
            #     raise ValueError("Vehicle should not stop at red light!")
        assert env.vehicle.speed < 0.1

        # move
        test_success = False
        for s in range(1, 1000):
            o, r, d, t, i = env.step([0, 1])
            if i["arrive_dest"]:
                test_success = True
                break
        stop_sign.destroy()
        assert test_success
    finally:
        env.close()


if __name__ == "__main__":
    # test_traffic_light_state_check(True, manual_control=False)
    # test_traffic_light_detection(True, manual_control=False)
    testBaseStopSign(True)
