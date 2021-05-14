from typing import Optional, Tuple

import cv2
from gym import Env, ObservationWrapper


class ImageWrapper(ObservationWrapper):
    """[summary]
    Image processing wrapper for gym enironment

    :param env: the environment
    :param grayscale: if True, then observation is grayscaled  
    :param normalize: if True, input are normalized 
    """
    def __init__(self, env: Env, grayscale: bool = True, normalize: bool = False):
        super().__init__(env)
        self.grayscale = grayscale
        self.normalize = normalize
    
    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[:, :, None]
     #       obs = obs
        if self.normalize:
            obs = obs / 255.0
        return obs
