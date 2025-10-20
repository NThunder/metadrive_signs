import numpy as np

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask, CollisionGroup
from metadrive.constants import MetaDriveType, Semantics
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.pg.utils import generate_static_box_physics_body


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

    _MODEL = None  # Shared model to avoid reloading

    def __init__(
        self,
        lane,
        position=None,
        name=None,
        random_seed=None,
        config=None,
        escape_random_seed_assertion=False,
        show_model=True,
    ):
        super(BaseStopSign, self).__init__(name, random_seed, config, escape_random_seed_assertion)
        self.set_metadrive_type(MetaDriveType.TRAFFIC_SIGN)
        self.lane = lane
        self._show_model = show_model

        # Create a thin invisible collision box (for LiDAR / detection)
        self.lane_width = lane.width_at(0) if lane else 4.0
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
        if position is None:
            position = lane.position(self.PLACE_LONGITUDE, -self.lane_width / 2 - 0.5)  # Slightly off to the side
        self.set_position(position, self.SIGN_HEIGHT / 2)
        self.set_heading_theta(lane.heading_theta_at(self.PLACE_LONGITUDE) + np.pi / 2)  # Face the road

        # Load and attach visual model
        if self.render and self._show_model:
            if BaseStopSign._MODEL is None:
                model_path = AssetLoader.file_path("models", "traffic_sign", "stop_sign.gltf")
                BaseStopSign._MODEL = self.loader.loadModel(model_path)
                BaseStopSign._MODEL.setPos(0, 0, self.SIGN_HEIGHT)
                BaseStopSign._MODEL.setH(-90)
                BaseStopSign._MODEL.hide(CamMask.Shadow)
            self._visual_model = BaseStopSign._MODEL.instanceTo(self.origin)
            self.origin.setScale(1.0)

    def destroy(self):
        if hasattr(self, '_visual_model'):
            self._visual_model.detachNode()
        super(BaseStopSign, self).destroy()
        self.lane = None

    @property
    def top_down_color(self):
        return [255, 0, 0]  # Red for stop sign

    @property
    def top_down_width(self):
        return self.SIGN_WIDTH

    @property
    def top_down_length(self):
        return self.SIGN_DEPTH

    @property
    def LENGTH(self):
        return self.SIGN_DEPTH

    @property
    def WIDTH(self):
        return self.SIGN_WIDTH