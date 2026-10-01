from dataclasses import replace
from .shared_agent_v2 import ALGORITHM as _SHARED
ALGORITHM = replace(_SHARED, key="shared-agent-no-preload@1", label="Shared sandbox, no preload", aliases=("shared-no-preload",), default_env_pool_factor=1)
