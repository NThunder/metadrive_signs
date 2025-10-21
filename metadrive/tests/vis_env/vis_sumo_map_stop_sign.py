"""use netconvert --opendrive-files CARLA_town01.net.xml first"""
import logging

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs import BaseEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.utils.pg.utils import ray_localization

from metadrive.envs.top_down_env import TopDownMetaDrive

from metadrive.component.traffic_sign.base_stop_sign import BaseStopSign

class SimpleTrafficManager(BaseManager):
    """
    A simple traffic creator, which creates one vehicle to follow a specified route with IDM policy.
    """
    def __init__(self):
        super(SimpleTrafficManager, self).__init__()
        self.generated_v = None
        self.arrive_dest = False

    def after_reset(self):
        """
        Create vehicle and use IDM for controlling it. When there are objects in front of the vehicle, it will yield
        """
        self.arrive_dest = False
        path_to_follow = []
        # print("Available lane IDs:")
        # for lane_id in self.engine.current_map.road_network.graph.keys():
        #     print(lane_id)
        for lane_index in ["lane_:2_0_0", "lane_:3_0_0", "lane_-43_1"]:
            path_to_follow.append(self.engine.current_map.road_network.get_lane(lane_index).get_polyline())
        path_to_follow = np.concatenate(path_to_follow, axis=0)
        self.generated_v = self.spawn_object(
            SVehicle, vehicle_config=dict(), position=path_to_follow[0], heading=-np.pi
        )
        TrajectoryIDMPolicy.NORMAL_SPEED = 20
        self.add_policy(
            self.generated_v.id,
            TrajectoryIDMPolicy,
            control_object=self.generated_v,
            random_seed=0,
            traj_to_follow=PointLane(path_to_follow, 2)
        )

    def before_step(self):
        """
        When arrive destination, stop
        """
        policy = self.get_policy(self.generated_v.id)
        if policy.arrive_destination:
            self.arrive_dest = True

        if not self.arrive_dest:
            action = policy.act(do_speed_control=True)
        else:
            action = [0., -1]
        self.generated_v.before_step(action)  # set action


class MyEnv(BaseEnv):
    def reward_function(self, agent):
        """Dummy reward function."""
        return 0, {}

    def cost_function(self, agent):
        """Dummy cost function."""
        return 0, {}

    def done_function(self, agent):
        """Dummy done function."""
        return False, {}

    def get_single_observation(self):
        """Dummy observation function."""
        return DummyObservation()

    def setup_engine(self):
        """Register the map manager"""
        super().setup_engine()
        map_path = AssetLoader.file_path("carla", "map_2.net.xml", unix_style=False)
        self.engine.register_manager("map_manager", SumoMapManager(map_path))
        self.engine.register_manager("traffic_manager", SimpleTrafficManager())


if __name__ == "__main__":
    # create env
    env = MyEnv(
        dict(
            use_render=False,
            vehicle_config={"spawn_position_heading": [(-347.5359589,  -117.64674795), 0]},
            manual_control=True,  # we usually manually control the car to test environment
            # use_mesh_terrain=True,
            log_level=logging.CRITICAL,
            window_size =  (1200, 800),
            show_coordinates = True,
        )
    )  # suppress logging message
    env.reset()

    lane = env.engine.current_map.road_network.get_lane("lane_-43_1")
    longitudinal_pos = lane.length - 0.5  # за 0.5 м до конца полосы
    
    stop_sign = env.engine.spawn_object(
        BaseStopSign,
        lane=lane,
        longitudinal_offset=-3.0,  # за 3 метра до конца полосы
        lateral_offset=lane.width_at(0) / 2 + 0.8  # справа от дороги
    )

    has_entered_approach_zone = False
    has_stopped_before_line = False
    violation_recorded = False
    # Зона подхода: за 8 метров до стоп-линии
    APPROACH_DISTANCE = 8.0

    for i in range(10000):
        obs, reward, termination, truncate, info, = env.step(env.action_space.sample())
        
        if env.agent.crash_vehicle:
            print("💥 Столкнулся с автомобилем!")
        # current_lanes = ray_localization(
        #     heading=env.agent.heading,
        #     position=env.agent.position,
        #     engine=env.engine,
        #     use_heading_filter=True
        # )
        # if current_lanes:
        #     closest_lane_info = current_lanes[0]  # (lane, long, lat, distance)
        #     current_lane = closest_lane_info[0]
        #     print("Текущая полоса:", current_lane.index)
        # else:
        #     current_lane = None
        #     print("Не удалось определить полосу")

        vehicle = env.agent
        lane = stop_sign.lane

        if lane is None or stop_sign.stop_line_longitudinal_position is None:
            continue

        veh_long = lane.local_coordinates(vehicle.position)[0]
        stop_long = stop_sign.stop_line_longitudinal_position

        # 1. Проверяем, вошёл ли автомобиль в зону подхода
        if not has_entered_approach_zone and (stop_long - veh_long) <= APPROACH_DISTANCE and veh_long < stop_long:
            has_entered_approach_zone = True
            print("🚦 В зоне подхода к стоп-линии")

        # 2. Если в зоне — проверяем остановку ДО линии
        if has_entered_approach_zone and not has_stopped_before_line and not violation_recorded:
            if vehicle.speed < 0.1 and veh_long < stop_long:
                has_stopped_before_line = True
                print(f"✅ Остановился перед стоп-линией! Скорость: {vehicle.speed:.2f} m/s")

        # 3. Проверяем, пересёк ли стоп-линию
        if not violation_recorded and veh_long >= stop_long:
            if has_stopped_before_line:
                print("👍 Корректно проехал после остановки")
            else:
                violation_recorded = True
                print(f"❌ НАРУШЕНИЕ! Проезд без остановки! Скорость: {vehicle.speed:.2f} m/s")

        env.render(mode="top_down", text={
            "Approach": has_entered_approach_zone,
            "Stopped": has_stopped_before_line,
            "Violation": violation_recorded,
            "Veh_long": f"{veh_long:.1f}",
            "Stop_long": f"{stop_long:.1f}"
        }, film_size=(4000, 4000))

        if termination:
            if info["arrive_dest"]:
                print("🎯 Успех! Доехал до цели.")
            elif info["out_of_road"]:
                print("⚠️ Выехал за пределы дороги!")
            elif info.get("crash", False):
                print("💥 Произошло столкновение!")
            elif info["max_step"]:
                print("⏳ Закончилось время эпизода.")
            else:
                print("⏹️ Эпизод завершён по другой причине.")
            break
    env.close()
