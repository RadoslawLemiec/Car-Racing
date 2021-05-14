from stable_baselines3.common.callbacks import BaseCallback


class EpisodicCallback(BaseCallback):
    """
    Calculate episodical rewards for many environments.

    :param envs_num: number of parrarel environment
    :param reward_normalizaion: if True record original reward instead of normalized one
    :param verbose: verbosity level: 0 - none, 1 - training info, 2 - debug
    """
    def __init__(self, envs_num: int = 1, reward_normalizaion: bool = False, verbose:int = 1):
        super().__init__(verbose)
        self._reward_sums = [0.0] * envs_num
        self.reward_normalizaion = reward_normalizaion

    def _on_step(self) -> bool:
        """ Call after each `env.step()`
        :return: if the callback returns False, training is aborted early
        """
        rewards = self.model.env.get_original_reward() if self.reward_normalizaion else self.locals["rewards"]
        # sum rewards for current step with _reward_sums
        self._reward_sums = [sum(x) for x in zip(self._reward_sums, rewards)] 
        # record episodic reward for every environment in terminal state
        for done, reward_sum in zip(self.locals["dones"], self._reward_sums):
            if done:
                self.logger.record("episodic_reward", reward_sum)
        # zero rewards for enviorment in terminal state
        self._reward_sums = [reward_sum * (1-is_done) for reward_sum, is_done in zip(self._reward_sums, self.locals["dones"])]         
        return True
