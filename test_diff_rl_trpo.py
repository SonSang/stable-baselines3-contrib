from diff_on_policy_algorithms.diff_trpo import DiffTRPO
from diff_on_policy_algorithms.diff_policies import DiffActorCriticPolicy
from test_argparserd import TestParser

parser = TestParser("TRPO")
print("Sample Enhancement: {} / Policy Enhancement: {}".format(parser.args.se, parser.args.pe))

model = DiffTRPO(DiffActorCriticPolicy, 
                parser.env, 
                verbose=1, 
                device='cpu',
                # n_steps=8,
                # batch_size=8,
                sample_enhancement=parser.args.se, 
                policy_enhancement=parser.args.pe,
                debug=False)

model.learn(total_timesteps=parser.args.n_steps, 
            eval_env=parser.eval_env, 
            eval_freq=parser.args.n_eval_steps, 
            n_eval_episodes=20, 
            eval_log_path=parser.log_path)

model.save(path=parser.log_path + "/final_model.zip")
model.load(parser.log_path + "/best_model.zip", device="cpu")

parser.record_video(model)
parser.close()		# Should be called for saving video