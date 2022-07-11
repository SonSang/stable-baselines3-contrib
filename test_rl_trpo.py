from sb3_contrib import TRPO
from test_argparser import TestParser

parser = TestParser("TRPO")

model = TRPO("MlpPolicy", parser.env, verbose=1, device='cpu')
model.learn(
	total_timesteps=parser.args.n_steps, 
	eval_env=parser.eval_env, 
	eval_freq=parser.args.n_eval_steps, 
	n_eval_episodes=20, 
	eval_log_path=parser.log_path
)

model.save(path=parser.log_path + "/final_model.zip")
model.load(parser.log_path + "/best_model.zip", device="cpu")

parser.record_video(model)
parser.close()		# Should be called for saving video