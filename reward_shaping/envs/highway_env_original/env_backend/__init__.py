# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# Import the envs module so that envs register themselves
import reward_shaping.envs.highway_env_original.env_backend.env
