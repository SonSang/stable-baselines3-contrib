from diff_on_policy_algorithms.diff_trpo import DiffTRPO
from diff_on_policy_algorithms.diff_policies import DiffActorCriticPolicy
from sb3_contrib import TRPO
from compare_argparser import TestParser

parser = TestParser("TRPO")
print("Sample Enhancement: {} / Policy Enhancement: {}".format(parser.args.se, parser.args.pe))

n_eval_episodes = 20

# Base
model = TRPO("MlpPolicy", parser.b_env, verbose=0, device='cpu')
model.save(parser.log_path_base + "/init_model.zip")
model.learn(
	total_timesteps=parser.args.n_steps, 
	eval_env=parser.b_eval_env, 
	eval_freq=parser.args.n_eval_steps, 
	n_eval_episodes=n_eval_episodes, 
	eval_log_path=parser.log_path_base
)
model.save(path=parser.log_path_base + "/final_model.zip")
model.set_parameters(parser.log_path_base + "/best_model.zip", device="cpu")

if not parser.args.no_record:
    parser.record_video(model, True)
parser.close(True)

# Diff
model = DiffTRPO(DiffActorCriticPolicy, 
                parser.d_env, 
                verbose=0, 
                device='cpu',
                # n_steps=8,
                # batch_size=8,
                sample_enhancement=parser.args.se, 
                policy_enhancement=parser.args.pe,
                debug=False)
# For fair comparison, use the same initial model parameters
model.set_parameters(parser.log_path_base + "/init_model.zip", device="cpu")
model.save(parser.log_path_diff + "/init_model.zip")

model.learn(total_timesteps=parser.args.n_steps, 
            eval_env=parser.d_eval_env, 
            eval_freq=parser.args.n_eval_steps, 
            n_eval_episodes=n_eval_episodes, 
            eval_log_path=parser.log_path_diff)

model.save(path=parser.log_path_diff + "/final_model.zip")
model.set_parameters(parser.log_path_diff + "/best_model.zip", device="cpu")

if not parser.args.no_record:
    parser.record_video(model, False)
parser.close(False)		# Should be called for saving video