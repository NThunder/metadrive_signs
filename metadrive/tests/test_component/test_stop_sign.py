from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.component.traffic_sign.base_stop_sign import BaseStopSign
import numpy as np

import random

import matplotlib.pyplot as plt

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import HELP_MESSAGE
from metadrive.examples.ppo_expert.numpy_expert import expert

def point_line_side(p, a, b):
    """Возвращает знак: с какой стороны точки p относительно направленной линии a->b"""
    return np.cross(b - a, p - a)

def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        # print("Drawing {}-th semantic map!".format(count))
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()


def has_crossed_stop_line(vehicle, stop_sign):
    """
    Проверяет, пересёк ли автомобиль стоп-линию ЗА знаком (в направлении движения).
    """
    lane = stop_sign.lane
    vehicle_long = lane.local_coordinates(vehicle.position)[0]  # продольная координата на полосе
    stop_long = stop_sign.stop_line_longitudinal_position
    return vehicle_long >= stop_long


# STOP_SIGN_POSITION = np.array([49.57184600830078, -2.373173952102661])

# ROAD_DIRECTION = np.pi / 2  # ← ПОМЕНЯЙ ЭТО, ЕСЛИ НАПРАВЛЕНИЕ ДРУГОЕ!

# LINE_NORMAL = np.array([np.cos(ROAD_DIRECTION), np.sin(ROAD_DIRECTION)])

# STOP_LINE_D = np.dot(LINE_NORMAL, STOP_SIGN_POSITION)

# HALF_LINE_LENGTH = 3.0  # 3 метра в каждую сторону → итого 6 м

# # Вектор ВДОЛЬ стоп-линии (перпендикулярен направлению дороги)
# line_direction = np.array([-np.sin(ROAD_DIRECTION), np.cos(ROAD_DIRECTION)])  # поворот на 90°

# point_a = STOP_SIGN_POSITION + line_direction * HALF_LINE_LENGTH
# point_b = STOP_SIGN_POSITION - line_direction * HALF_LINE_LENGTH


def testBaseStopSign(render=False, debug=False):
    env = TopDownMetaDrive(
        {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "manual_control": True,
            "use_render": False,
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
        lane_before_intersection = env.current_map.road_network.graph[">>"][">>>"][2]
        lane = env.current_map.road_network.graph[">>"][">>>"][2]
        longitudinal_pos = lane.length - 0.5  # за 0.5 м до конца полосы
        # Центр линии — на оси полосы
        center = np.array(lane.position(longitudinal_pos, 0.0))

        # Направление движения в этой точке
        heading = lane.heading_theta_at(longitudinal_pos)
        
        # Вектор ПОПЕРЁК дороги (влево от направления движения)
        perpendicular = np.array([-np.sin(heading), np.cos(heading)])

        # Ширина линии = ширина дороги + запас
        half_width = lane.width_at(longitudinal_pos) / 2 + 0.5
        
        point_a = center + perpendicular * half_width
        point_b = center - perpendicular * half_width
        
        stop_sign = env.engine.spawn_object(
            BaseStopSign,
            lane=lane_before_intersection,
            longitudinal_offset=-3.0,  # за 3 метра до конца полосы
            lateral_offset=lane_before_intersection.width_at(0) / 2 + 0.8  # справа от дороги
            # position=STOP_SIGN_POSITION,
        )
    

        # stop_sign1 = env.engine.spawn_object(
        #     BaseStopSign,
        #     lane=None,

        #     position=point_a,
        # )
        
        # stop_sign2 = env.engine.spawn_object(
        #     BaseStopSign,
        #     lane=None,
        #     position=point_b,
        # )


        # Флаги для логики
        has_stopped_before = False
        violation = False
        for step in range(2000):
            # lane = env.vehicle.lane  # текущая полоса, на которой находится автомобиль
            # print("Текущая полоса:", lane.index)
            if render:
                
                color = list([1, 0, 0])
                env.engine.draw_line_2d(
                    start_p=point_a.tolist(), 
                    end_p=point_b.tolist(),
                    color=color,                     # красный
                    thickness=3
                )
            
            o, r, d, t, info = env.step([0, 0.1])  # [steering, throttle]
            env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))

            if step % 50 == 0:
                draw_multi_channels_top_down_observation(o, show_time=5) 

            has_stopped = False
            violation = False
            
            if not has_stopped and env.vehicle.speed < 0.1:
                if not has_crossed_stop_line(env.vehicle, stop_sign):
                    has_stopped = True
                    # print("✅ Остановился перед стоп-линией")

            if not violation and has_crossed_stop_line(env.vehicle, stop_sign):
                if not has_stopped:
                    violation = True
                    print("❌ НАРУШЕНИЕ!")

            if d:
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

        stop_sign.destroy()

    finally:
        env.close()

if __name__ == "__main__":
    testBaseStopSign(True, False)