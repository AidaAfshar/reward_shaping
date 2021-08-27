import argparse as parser

from reward_shaping.training.train import train


def main(args):
    train_params = {'steps': args.steps,
                    'video_every': int(args.steps/10),
                    'n_recorded_episodes': 5,
                    'eval_every': int(args.steps/10),
                    'n_eval_episodes': 10,
                    'checkpoint_every': int(args.steps/10)}
    train(args.env, args.task, args.reward, train_params, seed=args.seed)


if __name__ == "__main__":
    envs = ['cart_pole_obst', 'bipedal_walker']
    parser = parser.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=envs)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
