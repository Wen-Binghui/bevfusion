# from mmcv.runner.hooks import HOOKS, Hook 
# import numbers	
# from mmcv.runner.dist_utils import master_only

# __all__ = ["WandbLoggerHook"]

# @HOOKS.register_module() 	
# class WandbLoggerHook(HOOKS):
	
#     def __init__(self,	
#                  log_dir=None,	
#                  interval=10,	
#                  ignore_last=True,	
#                  reset_flag=True):	
#         super(WandbLoggerHook, self).__init__(interval, ignore_last,	
#                                               reset_flag)	
#         self.import_wandb()
	
    
#     def import_wandb(self):	
#         try:	
#             import wandb	
#         except ImportError:
#             raise ImportError(	
#                 'Please run "pip install wandb" to install wandb')	
#         self.wandb = wandb
	
#     @master_only
#     def before_run(self, runner):	
#         if self.wandb is None:	
#             self.import_wandb()	
#         self.wandb.init()	

#     @master_only
#     def log(self, runner):	
#         metrics = {}	
#         for var, val in runner.log_buffer.output.items():	
#             if var in ['time', 'data_time']:	
#                 continue	
#             tag = '{}/{}'.format(var, runner.mode)	
#             runner.log_buffer.output[var]	
#             if isinstance(val, numbers.Number):	
#                 metrics[tag] = val	
#         if metrics:	
#             self.wandb.log(metrics, step=runner.iter)

#     @master_only	
#     def after_run(self, runner):	
#         self.wandb.join()