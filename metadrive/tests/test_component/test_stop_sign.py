from metadrive.component.traffic_sign.base_stop_sign import BaseStopSign

def test_stop_sign(render=True):
    from metadrive.envs.metadrive_env import MetaDriveEnv
    env = MetaDriveEnv({
        "map": "X",
        "use_render": render,
        "manual_control": True,
        "vehicle_config": {"show_lidar": True}
    })
    env.reset()
    try:
        lane = env.current_map.road_network.graph[">>>"]["1X1_0_"][0]
        stop_sign = env.engine.spawn_object(BaseStopSign, lane=lane)
        for _ in range(500):
            env.step([0, 1])
        stop_sign.destroy()
    finally:
        env.close()

if __name__ == "__main__":
    test_stop_sign(render=True)