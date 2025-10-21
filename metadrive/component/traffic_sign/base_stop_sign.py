import numpy as np

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask, CollisionGroup
from metadrive.constants import MetaDriveType, Semantics
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.pg.utils import generate_static_box_physics_body

from panda3d.core import CardMaker, NodePath, Texture
from panda3d.core import Vec3, ShaderTerrainMesh, Texture, TextureStage, Shader, Filename


import os

from panda3d.core import SamplerState, PNMImage, CardMaker, LQuaternionf, NodePath

def draw_stop_text_on_road(engine, position, size=2.0):
    # 1. Загружаем текстуру
    from metadrive.engine.asset_loader import AssetLoader
    import os
    texture_path = AssetLoader.file_path("textures", "stop_text.png")
    
    if not os.path.exists(texture_path):
        print("⚠️ stop_text.png не найден!")
        return None

    texture = engine.loader.loadTexture(texture_path)
    texture.set_format(Texture.F_srgb)  # как в terrain
    texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    texture.setMagfilter(SamplerState.FT_linear)
    texture.setAnisotropicDegree(8)

    # 2. Создаём CardMaker
    cm = CardMaker("stop_sign_card")
    cm.setFrame(-size/2, size/2, -size/2, size/2)
    cm.setUvRange((0, 0), (1, 1))  # ← ВАЖНО: UV от 0 до 1

    card_np = NodePath(cm.generate())
    card_np.setPos(position[0], position[1], 0.01)
    card_np.setHpr(90, -90, 0)  # лежит на земле

    # 3. Создаём TextureStage (как в terrain)
    ts_color = TextureStage("stop_sign_color")
    ts_color.set_mode(TextureStage.M_modulate)  # стандартный режим наложения

    # 4. Применяем текстуру
    card_np.setTexture(ts_color, texture)
    card_np.setLightOff(1)  # отключаем освещение

    # 5. Добавляем в сцену
    card_np.reparentTo(engine.render)
    return card_np

class BaseStopSign(BaseObject):
    """
    A static stop sign placed near intersections.
    It acts as a detectable object and can enforce stopping behavior.
    """
    SEMANTIC_LABEL = Semantics.TRAFFIC_SIGN.label
    SIGN_HEIGHT = 2.0  # Height above ground
    SIGN_WIDTH = 0.6
    SIGN_DEPTH = 0.1
    PLACE_LONGITUDE = 5.0  # Distance along the lane to place the sign
    
    STOP_SIGN_MODEL = {}

    def __init__(
        self,
        lane,
        position=None,
        name=None,
        random_seed=None,
        config=None,
        escape_random_seed_assertion=False,
        show_model=True,
        longitudinal_offset=-2.0,  # Знак за 2 метра ДО начала перекрёстка (отрицательное смещение)
        lateral_offset=None,       # Смещение вбок: положительное = вправо по ходу движения
    ):
        super(BaseStopSign, self).__init__(name, random_seed, config, escape_random_seed_assertion)
        self.set_metadrive_type(MetaDriveType.TRAFFIC_SIGN)
        self.lane = lane
        self._show_model = show_model

        # Create a thin invisible collision box (for LiDAR / detection)
        self.lane_width = lane.width_at(0) if lane else 4.0
        
        if lateral_offset is None:
            # Ставим знак справа от полосы (вне проезжей части)
            lateral_offset = self.lane_width / 2 + 0.8  # 0.8 м — отступ от края дороги
        
        collision_body = generate_static_box_physics_body(
            self.SIGN_DEPTH,
            self.SIGN_WIDTH,
            self.SIGN_HEIGHT,
            object_id=self.id,
            type_name=MetaDriveType.TRAFFIC_SIGN,
            ghost_node=True,
        )
        self.add_body(collision_body, add_to_static_world=False)  # Add to dynamic world for detection

        # Determine position
       # Определяем позицию
        if position is None:
            # Берём точку на полосе с продольным смещением (например, -2.0 = за 2 м до начала участка)
            long_pos = max(0.0, lane.length - abs(longitudinal_offset)) if longitudinal_offset < 0 else longitudinal_offset
            # Но лучше — ставить относительно КОНЦА полосы, если это подъезд к перекрёстку
            # В MetaDrive перекрёстки часто в конце полосы, поэтому:
            long_pos = lane.length + longitudinal_offset  # если longitudinal_offset = -2, то за 2 м до конца
            long_pos = max(0.1, min(lane.length - 0.1, long_pos))  # clamp

            # Поперечное смещение: положительное = вправо по направлению движения
            position = lane.position(long_pos, lateral_offset)

        draw_stop_text_on_road(self.engine, position)
        self.set_position(position, 0)  # z = половина высоты (2.0 / 2)

        # Поворачиваем знак, чтобы он "смотрел" на дорогу (перпендикулярно направлению движения)
        # heading = lane.heading_theta_at(long_pos)
        # self.set_heading_theta(heading + np.pi / 2)  # поворот на 90° вправо → знак лицом к дороге

        # Load and attach visual model
        if self.render:
            if "stop" not in  BaseStopSign.STOP_SIGN_MODEL and self._show_model:
                model_path = AssetLoader.file_path("models", "traffic_sign", "stop_sign.gltf")
                model = self.loader.loadModel(model_path)
                model.setPos(0, 0, 2.0)
                model.setH(-90)
                model.hide(CamMask.Shadow)
                BaseStopSign.STOP_SIGN_MODEL["stop"] = model  # сохраняем в словарь

            self._visual_model = BaseStopSign.STOP_SIGN_MODEL["stop"].instanceTo(self.origin)
    
    @property
    def stop_line_longitudinal_position(self):
        # Стоп-линия за 0.5 м до конца полосы
        return self.lane.length - 0.5

    def destroy(self):
        if hasattr(self, '_visual_model'):
            self._visual_model.detachNode()
        super(BaseStopSign, self).destroy()
        self.lane = None

    @property
    def top_down_length(self):
        return self.RADIUS * 4

    @property
    def top_down_width(self):
        return self.RADIUS * 4

    @property
    def top_down_color(self):
        return [255, 0, 0]

    @property
    def LENGTH(self):
        return self.SIGN_DEPTH

    @property
    def WIDTH(self):
        return self.SIGN_WIDTH