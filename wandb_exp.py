import wandb
from utils import generate_neat_config
import evolve as evolve

wandb.login()

def objective(run, n_seasons, seed, n_gens, energy_costs):
    winner_fitness = evolve.run(wandb_run=run , n_seasons=n_seasons, seed=seed, n_gens=n_gens, energy_costs=energy_costs)
    return winner_fitness


def main(args):

    run_id = args.run_id
    if run_id is None:
        run_id = str(args.n_seasons) + "s" + str(args.seed)
        if args.EC:
            run_id += "_EC"
        else:
            run_id += "_NEC"

    

    run = wandb.init(
            project=args.project,
            id=args.run_id,
            #resume="must"
            )

    #run.save("neat_config_exp_random_fitness_v2")

    winner_fitness = objective(run, args.n_seasons, args.seed, args.n_gens, args.EC) 
    wandb.log({"winner_fitness": winner_fitness})

    # Finish W&B run
    run.finish()
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')

     # project name
    parser.add_argument('--project', type=str, default="ECNCECE",
                        help='Wandb project name')

    # id name
    parser.add_argument('--run_id', type=str, default=None,
                        help='seed value')

    # seed argument
    parser.add_argument('--seed', type=int, default=1361,
                        help='seed value')

    # n_seasons argument
    parser.add_argument('--n_seasons', type=int, default=4,
                        help='Number of seasons in environment')

    # n_gens argument
    parser.add_argument('--n_gens', type=int, default=1,
                        help='Number of Generations')

    # EC argument
    parser.add_argument('--EC', action='store_true', help='Impose energy costs that scale with ANN size')


    args = parser.parse_args()
    print("Argument values:")
    print("Wandb Project Name: ", args.project)
    print("Wandb Project Run ID: ", Pargs.run_id)
    print("seed: ", args.seed)
    print("n_seasons: ", args.n_seasons)
    print("Generations: ", args.n_gens)
    print("Energy Costs: ", args.EC)

    main(args)


