
import neat
#from env_v4 import MyEnv
#import numpy as np
import networkx as nx
#import visualize
#import matplotlib.pyplot as plt
#import agent_jax_WBinit as agent
import agent_jaxv9 as agent
import utils
from custom_genome_v2 import CustomGenome
#import random
from flax import linen as nn

import multiprocessing
#import os

#from memory_profiler import profile
import gc

#import tracemalloc
import jax

from CustomReporter import CustomReporter
from parallel_v2 import ParallelEvaluator

import register
#tracemalloc.start()
from stable_baselines3.common.env_util import make_vec_env
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')

#from forward.models import ForWard as fwd
#env = MyEnv(n_seasons=4, col_dist=True, v=4)

#env_args = {'n_seasons': 4, 'col_dist': True, 'v': 4, 'size': 20}

def log_complexity(genome, config, wandb_run, save=False, winner=False, gen=""):
        pruned = genome.get_pruned_copy(config.genome_config)

        G = utils.make_graph(pruned, config, 243)

        if save:
            run_name = wandb_run.id
            filename = run_name + "_gen" + str(gen) + "_Graph.adjlist"
            nx.write_adjlist(G, filename)
            wandb_run.save(filename)
            filename = run_name + "_gen" + str(gen) + "_Graph.png"
            utils.show_layered_graph(G, save=True, filename=filename)
            wandb_run.save(filename)

        ns = utils.get_ns(G)
        nc, mod, glob_eff = utils.get_nc(G)
        num_nodes = G.number_of_nodes()
        num_conn = G.number_of_edges()

        #if winner:
        #    wandb_run.log({"winner_ns": ns, "winner_nc": nc})
        #else:
        #    wandb_run.log({"gen_best_ns": ns, "gen_best_nc": nc, "gen_best_num_nodes": num_nodes, "gen_best_num_conn": num_conn})
        return ns, nc, num_nodes, num_conn, mod, glob_eff


#@profile
def eval_genome(genome, config, learn=True, total_timesteps=100_000, skip_evaluated=True, n_seasons=2, seed=11108):
    #global env
    if skip_evaluated and genome.fitness is not None:
        return genome.fitness
    #env = MyEnv(n_seasons=4, col_dist=True, v=4, size=20)#, shock_obs=False, v=1)
    #env.reset(seed=1361)
    #env_args = {'n_seasons': 3, 'col_dist': True, 'v': 4, 'size': 20}
    #env = make_vec_env("Env-energy", n_envs=5, env_kwargs=env_args)

    input_size = 243 #env.observation_space.shape[0]
    #G = genome.graph
    pruned = genome.get_pruned_copy(config.genome_config)

    G = utils.make_graph(pruned, config, input_size)

    ns = utils.get_ns(G)

    connections = G.number_of_edges()
    if connections == 0:
        return -(ns*0.01) - 1

    energy_coef = 0.01/254
    env_args = {'n_seasons': n_seasons, 'col_dist': True, 'v': 4, 'size': 20, 'ns': ns, 'col_seed':seed, 'col_var':0.2, 'energy_coef':energy_coef}
    env = make_vec_env("Env-energy-v2", n_envs=5, env_kwargs=env_args)

    if genome.activation_fn == "relu":
        activation_fn = nn.relu
    else:
        activation_fn = nn.tanh

    learning_rate = genome.learning_rate
    if genome.lr_schedule == "linear":
         learning_rate = utils.linear_schedule(learning_rate)
   
    #try:
    if True:
        model = agent.build_model(env, G, seed=seed, lr=learning_rate, gamma=genome.gamma, batch_size=genome.batch_size,
                                n_steps=genome.n_steps, ent_coef=genome.ent_coef, clip_range=genome.clip_range, 
                                n_epochs=genome.n_epochs, gae_lambda=genome.gae_lambda, max_grad_norm=genome.max_grad_norm, 
                                vf_coef=genome.vf_coef, activation_fn=activation_fn)
        if learn:
            model.learn(total_timesteps=total_timesteps, progress_bar=False)

        mean_reward, std_reward = agent.evaluate(model,deterministic=True, n_episodes=100, print_out=False)
    
    '''
    except Exception as error:
        print(error)
        return -100
    '''
    env.close()
    del model
    del G
    del env

    #jax.clear_caches()
    #gc.collect()
    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics('lineno')
    #for stat in top_stats[:5]:
    #    print(stat)
    #    print()
    return mean_reward

def eval_genomes(genomes, config):
    learn = False

    #i=1
    for genome_id, genome in genomes:
        #genome.fitness = -100
        print("genome ", genome_id)
        if genome.fitness is None:
            genome.fitness = eval_genome(genome, config, learn)
        #print()
        #print("i=",i ,", ns=", genome.ns, ", nc=", genome.nc, ", score=",genome.fitness)
        #print("n_steps:",genome.n_steps, ", batch_size:", genome.batch_size, ", gamma:", genome.gamma,
        #       ', lr:', genome.learning_rate, ', activation_fn:', genome.activation_fn)
        #print(i)
        #i += 1


def run(config_file='neat_config_test', parallel=False, wandb_run=None, restore=False, checkpoint_file=None):

    if restore:
        checkpoint = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        p = neat.Population(checkpoint.config)
        p.population = checkpoint.population
        p.species = checkpoint.species
        p.generation = checkpoint.generation
        config = checkpoint.config

    else:
        # Load configuration.
        config = neat.Config(CustomGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
    
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = CustomReporter()
    p.add_reporter(stats)

    if wandb_run:
        filename_prefix = wandb_run.name + '-'
    else:
        filename_prefix = 'neat-checkpoint-'
    checkpointer = neat.Checkpointer(generation_interval=50, time_interval_seconds=None, filename_prefix=filename_prefix)
    p.add_reporter(checkpointer)

    if parallel:
        print("cpu count: ", int(multiprocessing.cpu_count()/2))
        print("test: (pop=150, gen=100, iter=100_000): RL params, 30 threads, 30 cores, StdOutReporter(False), timeout=300 sec, clear jax cashes not with threads, online")
        #print("test (gpu): 10 threads, n_tasks=8, 4 cards, timeout=300sec, online")
        print("testing n_envs=5 for 100_000 iters")
        pe = ParallelEvaluator(30, eval_genome, timeout=600)
        #winner = p.run(pe.evaluate, 3)
        
        gen_start = p.generation
        print("gen start: ", gen_start)

        for generation in range(gen_start, 400):
            #print("test if same 4")
            #if True:
            try:
                gen_best = p.run(pe.evaluate, 1)
            #else:
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if isinstance(e, multiprocessing.context.TimeoutError):
                    print("Timeout occurred during evaluation.")
                print("Saving checkpoint...")
                checkpointer.save_checkpoint(
                    config=config,
                    population=p.population,
                    species_set=p.species,
                    generation=p.generation,
                )
                print("checkpoint saved")
                raise
            if wandb_run:
                gen_mean = stats.get_fitness_mean()
                #wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean})
                if (p.generation-1) % 100 == 0:
                    ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(gen_best, config, wandb_run, save=True, gen = p.generation-1)
                else:
                    ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(gen_best, config, wandb_run)
                gen_best_task_perf = gen_best.fitness + (ns/254)#1 #TODO for v16! take away if not using v15. change for energy experiments
                wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean, 
                    "gen_best_ns_v2": ns, "gen_best_nc_v2": nc, "gen_best_num_nodes": num_nodes, "gen_best_num_conn": num_conn,
                    "gen_best_modularity": mod, "gen_best_global_efficiency": glob_eff,
                    "gen_best_batch_size": gen_best.batch_size, "gen_best_n_steps": gen_best.n_steps, "gen_best_gamma": gen_best.gamma,
                    "gen_best_learning_rate": gen_best.learning_rate, "gen_best_ent_coef": gen_best.ent_coef, "gen_best_clip_range": gen_best.clip_range,
                    "gen_best_n_epochs": gen_best.n_epochs, "gen_best_gae_lambda": gen_best.gae_lambda, "gen_best_max_grad_norm": gen_best.max_grad_norm,
                    "gen_best_vf_coef": gen_best.vf_coef, "gen_best_activation_fn": gen_best.activation_fn, "gen_best_lr_schedule": gen_best.lr_schedule,
                    "gen_best_task_perf": gen_best_task_perf})

            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')
            #for stat in top_stats[:5]:
            #    print(stat)
            #    print()
                #if p.generation-1 == 99:
                #    wandb_run.log({"winner_fitness": gen_best.fitness})
            jax.clear_caches()
            gc.collect()
        #if wandb_run:
        #    wandb_run.finish()
            #print("Population: ", p.population)
        winner = gen_best

        checkpointer.save_checkpoint(
            config=config,
            population=p.population,
            species_set=p.species,
            generation=p.generation,
        )        


    else:
        print("test: non parallel, 20 envs")

        gen_start = p.generation
        print("gen start: ", gen_start)
        # Run until a solution is found.
        #winner = p.run(eval_genomes, 10)
        for generation in range(20):
            try:
            #if True:
                gen_best = p.run(eval_genomes, 1)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                '''
                print("Saving checkpoint...")
                checkpointer.save_checkpoint(
                    config=config,
                    population=p.population,
                    species_set=p.species,
                    generation=p.generation,
                )
                print("checkpoint saved")
                '''
            #print()
            #print("gen_best fitness: ", gen_best.fitness)
            #print("gen_mean fitness: ", gen_mean)
            #print()
            if wandb_run:
                gen_mean = stats.get_fitness_mean()
                if (p.generation-1 > 0) and (p.generation-1) % 100 == 0:
                    ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(gen_best, config, wandb_run, save=True, gen = p.generation-1)
                else:
                    ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(gen_best, config, wandb_run)
                wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean,
                    "gen_best_ns": ns, "gen_best_nc": nc, "gen_best_num_nodes": num_nodes, "gen_best_num_conn": num_conn,
                    "gen_best_batch_size": gen_best.batch_size, "gen_best_n_steps": gen_best.n_steps, "gen_best_gamma": gen_best.gamma,
                    "gen_best_learning_rate": gen_best.learning_rate, "gen_best_ent_coef": gen_best.ent_coef, "gen_best_clip_range": gen_best.clip_range,
                    "gen_best_n_epochs": gen_best.n_epochs, "gen_best_gae_lambda": gen_best.gae_lambda, "gen_best_max_grad_norm": gen_best.max_grad_norm,
                    "gen_best_vf_coef": gen_best.vf_coef, "gen_best_activation_fn": gen_best.activation_fn, "gen_best_lr_schedule": gen_best.lr_schedule})
                #wandb_run.log({"gen": p.generation-1, "gen_best_fitness": gen_best.fitness, "gen_mean_fitness": gen_mean})
            jax.clear_caches()
            gc.collect()
        winner = gen_best


    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.

    #print(winner.nodes.keys())
    #print(winner.nodes.values())
    
    #winner_fitness = eval_genome(winner, p.config, total_timesteps=100_000, skip_evaluated=False)

    if wandb_run:
        ns, nc, num_nodes, num_conn, mod, glob_eff = log_complexity(winner, config, wandb_run, save=True, winner=True, gen = p.generation-1)
        wandb_run.log({"winner_ns_v2": ns, "winner_nc_v2": nc})
        '''
        pruned = winner.get_pruned_copy(config.genome_config)

        G = utils.make_graph(pruned, config, 243)

        nx.write_adjlist(G, "Graph.adjlist")
        wandb_run.save("Graph.adjlist")

        winner_ns = utils.get_ns(G)
        winner_nc = utils.get_nc(G)
        winner_num_nodes = G.number_of_nodes()
        winner_num_conn = G.number_of_edges()

        wandb_run.log({"ns": winner_ns, "nc": winner_nc, "num_nodes": winner_num_nodes, "num_conn": winner_num_conn})

        utils.show_layered_graph(G, save=True)
        wandb_run.save("Graph.png")
        '''

    return winner.fitness

#import sys
#import threading


def restore():
    checkpoint = neat.Checkpointer.restore_checkpoint("brisk-sweep-7-42")
    p = neat.Population(checkpoint.config)
    p.population = checkpoint.population
    p.species = checkpoint.species
    p.generation = checkpoint.generation
    p.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(10, eval_genome, timeout=10*60)

    #winner = p.run(eval_genomes, n=3)
    winner = p.run(pe.evaluate, 2)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    #threading.settrace(trace_calls)
    run(parallel=False)
    #restore()
    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics('lineno')
    #for stat in top_stats[:10]:
    #    print(stat)


#run(parallel=True)
