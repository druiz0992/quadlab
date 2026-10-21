"""Create and remove models in a running Gazebo world (UserCommands services).

As with the world control service, replies from Python are unreliable, so
requests use a short timeout and the reply is not trusted; the model's
appearance in (or disappearance from) the world pose stream is the evidence.
"""

from typing import Any

CREATE_SERVICE = "/world/dronelab/create"
REMOVE_SERVICE = "/world/dronelab/remove"


class EntityService:
    """Spawn models from SDF text and remove them by name."""

    def __init__(self, node: Any, reply_timeout_ms: int = 100) -> None:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.entity_factory_pb2 import EntityFactory
        from gz.msgs10.entity_pb2 import Entity

        self._node, self._timeout = node, reply_timeout_ms
        self._factory, self._entity, self._boolean = EntityFactory, Entity, Boolean

    def spawn(self, sdf_text: str) -> None:
        request = self._factory()
        request.sdf = sdf_text
        self._node.request(
            CREATE_SERVICE, request, self._factory, self._boolean, self._timeout
        )

    def remove(self, model_name: str) -> None:
        request = self._entity()
        request.name = model_name
        request.type = self._entity.MODEL
        self._node.request(
            REMOVE_SERVICE, request, self._entity, self._boolean, self._timeout
        )
