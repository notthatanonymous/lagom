from algo import Algorithm

from lagom.experiment import Config
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster


class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = Algorithm(name='A2C')
        
        return algo


class ExperimentMaster(BaseExperimentMaster):
    def process_algo_result(self, config, result):
        assert result is None
        
    def make_configs(self):
        config = Config()
        
        ##########################
        # General configurations #
        ##########################
        # Whether to use GPU
        config.add_item(name='cuda', val=True)
        # Random seeds: generated by `np.random.randint(0, np.iinfo(np.int32).max, 5)`
        config.add_grid(name='seed', val=[144682090, 591442434, 1746958036, 338375070, 689208529])
        
        ############################
        # Algorithm configurations #
        ############################
        # Learning rate
        config.add_item(name='algo:lr', val=1e-3)
        # Discount factor
        config.add_item(name='algo:gamma', val=0.99)
        # Whether to use learning rate scheduler
        config.add_item(name='algo:use_lr_scheduler', val=False)
        
        ##############################
        # Environment configurations #
        ##############################
        # Environment ID
        # Note that better to run environment only one by one
        # because of specific settings, e.g. train:T, log-interval for fair benchmark curve
        config.add_item(name='env:id', val='CartPole-v1')
        # Flag for continuous or discrete control
        continuous = False
        # Whether to standardize the observation and reward by running average
        config.add_item(name='env:normalize', val=False)
        
        #########################
        # Engine configurations #
        #########################
        # Max training timesteps
        # Alternative: 'train:iter' for training iterations
        config.add_item(name='train:timestep', val=1e6) # recommended: 1e-6, i.e. 1M timesteps
        # Number of Segment per training iteration
        config.add_grid(name='train:N', val=[1, 16, 32, 64])
        # Number of timesteps per Segment
        config.add_item(name='train:T', val=5)
        
        # Evaluation: number of episodes to obtain average episode reward
        # We do not specify T, because the Engine.eval will automatically use env.T for complete episode
        config.add_item(name='eval:N', val=10)
        
        #######################
        # Agent configuration #
        #######################
        # Whether to standardize the discounted returns
        config.add_item(name='agent:standardize', val=False)
        # Gradient clipping with max gradient norm
        config.add_item(name='agent:max_grad_norm', val=0.5)
        # Coefficient for policy entropy loss
        config.add_item(name='agent:entropy_coef', val=0.01)
        # Coefficient for value loss
        config.add_item(name='agent:value_coef', val=0.5)
        # For Gaussian policy
        if continuous:
            # Min std threshould, avoid numerical instability
            config.add_item(name='agent:min_std', val=1e-6)
            # Use constant std; If use trainable std, put None
            config.add_item(name='agent:constant_std', val=None)
            # Whether to have state dependence for learning std
            config.add_item(name='agent:std_state_dependent', val=False)
            # Std parameterization: 'exp' or 'softplus'
            config.add_item(name='agent:std_style', val='exp')
        
        ##########################
        # Logging configurations #
        ##########################
        # Periodic interval to log and save information
        config.add_item(name='log:interval', val=50)
        # Directory to save loggings
        config.add_item(name='log:dir', val=f'logs/train_N/{config.config_settings["env:id"][0]}')
        
        # Auto-generate list of all possible configurations
        configs = config.make_configs()
        
        return configs
