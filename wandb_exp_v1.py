import wandb
from utils import generate_neat_config
import evolveV5 as evolve
#import jax

#jax.config.update('jax_disable_jit', True)  # Disable JIT globally

wandb.login()

def objective(run):
    winner_fitness = evolve.run(config_file='neat_config_exp_v2', parallel=True, wandb_run=run, restore=True, checkpoint_file="3s10301_a05d05_v16-360")
    return winner_fitness


def main():

    run = wandb.init(
            project="CS1B_exp_v1",
            #notes="assign random fitness to check if Nc grows with Ns, recurrent",
            #notes="energy_coeff=0.01/254(v16),random colour (env-energy-v2), fixed Ns (-243), new Nc calc (nc_v2): remove inactive input nodes, custom_genome_v2, agen_jaxv9, new node addition", 
            id="3s10301_a05d05_v16",
            #id = "random_fitness_1361_recurrent"#,
            resume="must"
            )

    #run.save("neat_config_exp_random_fitness_v2")

    winner_fitness = objective(run) 
    wandb.log({"winner_fitness": winner_fitness})

    # Finish W&B run
    run.finish()


if __name__ == '__main__':
    main()


