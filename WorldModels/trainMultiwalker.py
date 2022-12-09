"""
trainMultiwalker.py

Author:
    Joe Miceli
    Mohammed Adib Oumer

Description:
    Python file to calculate the optimal weights to use for each controller of the
    multiwalker environment, Optimization is performed using a CMA-ES optimization strategy
    in a single process (no threading/MPI forking)

usage:
    python trainMultiwalker.py -c configs/multiwalker.config


psuedocode: 
    initialize settings
        create controllers for each agent and an optimizer for each controller

    # (master) 
        # create test_env using the params from the trained RNN model

        # Start the clock
        start_time = int(time.time())
        # initialize the generation
        t = 0   # USE A BETTER VARIABLE
        # loop

            # generate seeds to use in setting the environment when training
            
            # Get the current weight estimates from each controller's optimizer and set the controller
            for es, controller in zip(ES_LIST, CONTROLLER_LIST):
                weights = es.ask()
                controller.set_model_params(weights)
            
            # (slave)/(worker)
            # Create a training environment using the params from the trained RNN model
            env = make_env(args=config_args, dream_env=False) # training in dreams not supported yet

            # Evaluate these controllers
            reward_list, t_list = simulate_multiple_controllers(CONTROLLER_LIST, env,
            train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
            
            # (master)
            # Extract some statistics from the simulations
            mean_time_step = int(np.mean(t_list)*100)/100. # get average time step
            max_time_step = int(np.max(t_list)*100)/100. # get average time step
            avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
            std_reward = int(np.std(reward_list)*100)/100. # get average time step

            r_max = int(np.max(reward_list)*100)/100.
            r_min = int(np.min(reward_list)*100)/100.

            # Get the elapsed time
            curr_time = int(time.time()) - start_time

            # Update each optimizer and get the current solution for each,
            # set each controller in the global list
            for es, controller in zip(ES_LIST, CONTROLLER_LIST):
                
                es.tell(reward_list)
                
                es_solution = es.result()

                model_params = es_solution[0] # best historical solution
                reward = es_solution[1] # best reward, should be the same for each controller
                curr_reward = es_solution[2] # best of the current batch, should be the same for each controller 
                
                controller.set_model_params(np.array(model_params).round(4))

                # store parameters for each controller
                with open(filename, 'wt') as out:
                    res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

                # Store the training history for this optimizer,
                # Note that all the histories are the same for each optimizer except for potentially rms_stdev()
                h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)
                history.append(h)   # Update the global variable
                with open(filename_hist, 'wt') as out:
                    res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))


            # Every few generations, conduct an evaluation
            if (t == 1):
                best_reward_eval = avg_reward   # Initialize the best reward on the first pass
            if (t % eval_steps == 0): # evaluate on actual task at hand
                prev_best_reward_eval = best_reward_eval

                for es, controller in zip(ES_LIST, CONTROLLER_LIST):
                    model_params_quantized = np.array(es.current_param()).round(4)
                    controller.set_model_params(np.array(model_params).round(4))


                # Evaluate using the test_env
                reward_eval_list = evaluate_batch(CONTROLLER_LIST, test_env, max_len=-1, test_seed=t)  # must be updated to support multiple controllers

                # Extract statistics from the evaluation
                reward_eval = np.mean(reward_eval_list)
                r_eval_std = np.std(reward_eval_list)
                r_eval_min = np.min(reward_eval_list)
                r_eval_max = np.max(reward_eval_list)
                e_h = (t, reward_eval, r_eval_std, r_eval_min, r_eval_max)
                eval_hist.append(e_h)
                with open(filename_eval_hist, 'wt') as out:
                    res = json.dump(eval_hist, out, sort_keys=False, indent=0, separators=(',', ':'))
                
                improvement = reward_eval - best_reward_eval

                for es, controller in zip(ES_LIST, CONTROLLER_LIST):      
                    model_params_quantized = np.array(es.current_param()).round(4)
                    model_params_list = model_params_quantized.tolist()
                    eval_log.append([t, reward_eval, model_params_list])   # Will need one of these lists per controller
                    # Store the params used for evaluation and the results from it
                    with open(filename_log, 'wt') as out:
                        res = json.dump(eval_log, out)  # Will need a file for each controller
                    # If this is the best evaluation we've seen, then keep these params
                    if (len(eval_log) == 1 or reward_eval > best_reward_eval):
                        best_reward_eval = reward_eval
                        best_model_params_eval = model_params_list
                    else:
                        if retrain_mode:
                            print("reset to previous best params, where best_reward_eval = {}".format(best_reward_eval))
                            es.set_mu(best_model_params_eval)    

                    # Store the best parameters for this controller after every evaluation
                    with open(filename_best, 'wt') as out:
                        res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
            
                # Store the history of best evaluations
                curr_time = int(time.time()) - start_time
                best_record = [t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval]
                history_best.append(best_record)
                with open(filename_hist_best, 'wt') as out:
                    res = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

            # Increment the generation
            t += 1

"""
from utils import PARSER
import numpy as np
import json
import os
from controller import make_controller, simulate_multiple_controllers
from es import CMAES, SimpleGA, OpenES, PEPG
from env import make_env
import time


class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result


def evaluate_batch(ctrl_list, test_env, test_seed=-1, num_test_episode=10, max_len=-1):
  # TODO: is this function really necessary? couldn't we just call simulate_multiple_controllers
  rewards_list, t_list = simulate_multiple_controllers(ctrl_list, test_env,
        train_mode=False, render_mode=False, num_episode=num_test_episode, seed=test_seed, max_len=max_len)  
  return rewards_list

def initialize_settings(args):   
    global FILEBASE, CONTROLLER_LIST, ES_LIST
    population = 8 # not currently used (see es.py) # population = num_worker * num_worker_trial 
    sigma_init = args.controller_sigma_init
    sigma_decay = args.controller_sigma_decay
    exp_name = args.exp_name
    env_name = args.env_name
    num_episode = args.controller_num_episode
    optimizer = args.controller_optimizer
    antithetic = args.controller_antithetic
    filedir = 'results/{}/{}/log/'.format(exp_name, env_name)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    FILEBASE = filedir+env_name+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)

    if (env_name == 'multiwalker_v9'):
        # Create the controllers 
        # TODO: make this support n-controllers
        controller0 = make_controller(args, id="ctrl_0")
        controller1 = make_controller(args, id="ctrl_1")
        controller2 = make_controller(args, id="ctrl_2")
        CONTROLLER_LIST = [controller0, controller1, controller2]

        num_params = controller0.param_count
        print("size of model", num_params)

        # Create an optimizer for each controller
        # TODO: make this support n-controllers
        if optimizer == 'ses':
            ses = PEPG(num_params,
                sigma_init=sigma_init,
                sigma_decay=sigma_decay,
                sigma_alpha=0.2,
                sigma_limit=0.02,
                elite_ratio=0.1,
                weight_decay=0.005,
                popsize=population)
            es0 = ses
            es1 = ses
            es2 = ses
        elif optimizer == 'ga':
            ga = SimpleGA(num_params,
                sigma_init=sigma_init,
                sigma_decay=sigma_decay,
                sigma_limit=0.02,
                elite_ratio=0.1,
                weight_decay=0.005,
                popsize=population)
            es0 = ga
            es1 = ga
            es2 = ga
        elif optimizer == 'cma':
            # cma = CMAES(num_params,
            #     sigma_init=sigma_init,
            #     popsize=population)
            # es0 = cma
            # es1 = cma
            # es2 = cma
            es0 = CMAES(num_params,
                sigma_init=sigma_init,
                popsize=population)
            es1 = CMAES(num_params,
                sigma_init=sigma_init,
                popsize=population)
            es2 = CMAES(num_params,
                sigma_init=sigma_init,
                popsize=population)                            
        elif optimizer == 'pepg':
            pepg = PEPG(num_params,
                sigma_init=sigma_init,
                sigma_decay=sigma_decay,
                sigma_alpha=0.20,
                sigma_limit=0.02,
                learning_rate=0.01,
                learning_rate_decay=1.0,
                learning_rate_limit=0.01,
                weight_decay=0.005,
                popsize=population)
            es0 = pepg
            es1 = pepg
            es2 = pepg
        else:
            oes = OpenES(num_params,
                sigma_init=sigma_init,
                sigma_decay=sigma_decay,
                sigma_limit=0.02,
                learning_rate=0.01,
                learning_rate_decay=1.0,
                learning_rate_limit=0.01,
                antithetic=antithetic,
                weight_decay=0.005,
                popsize=population)
            es0 = oes
            es1 = oes
            es2 = oes

        ES_LIST = [es0, es1, es2]

    else:
        print("[ERROR] THIS FILE ONLY SUPPORTS THE multiwalker_v9 ENVIRONMENT.")
        print("Your env is: {}".format(env_name))


def main(args):
  # global optimizer, num_episode, num_test_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode, env_name, exp_name, batch_mode, config_args

    # optimizer = args.controller_optimizer
    num_episode = args.controller_num_episode
    num_test_episode = args.controller_num_test_episode
    eval_steps = args.controller_eval_steps
    # num_worker = args.controller_num_worker
    # num_worker_trial = args.controller_num_worker_trial
    # antithetic = (args.controller_antithetic == 1)
    retrain_mode = (args.controller_retrain == 1)
    # cap_time_mode= (args.controller_cap_time == 1)
    # seed_start = args.controller_seed_start
    # env_name = args.env_name
    # exp_name = args.exp_name
    # batch_mode = args.controller_batch_mode
    # config_args = args

    initialize_settings(args)

    # (master)
    # Make a test env using the parameters from the trained RNN model
    test_env = make_env(args, dream_env=False, render_mode=False, load_model=True)

    # Start the clock
    start_time = int(time.time())

    # Create a seeder object to use to seed the env during training
    seeder = Seeder(args.controller_seed_start)

    # Initialize the generation and evaluation params
    gen = 0
    best_reward_eval = 0
    best_model_params_eval = None

    # Initialize dictionaries to that map controllers to lists of histories
    history_dict = {}
    eval_dict = {}
    for controller in CONTROLLER_LIST:
        # Initialize an empty lists
        history_dict[controller] = []   
        eval_dict[controller] = []

    # Initialize lists to store data for logging (these don't need to map to controllers 
    # because they don't contain controller-specific data)
    eval_hist = []
    history_best = []

    # File names that are not controller-specific can also be defined
    filename_eval_hist = FILEBASE+'.eval_hist.json'
    filename_hist_best = FILEBASE+'.hist_best.json'

    # max time steps in a simulation (-1 means ignore)
    max_len = -1 

    # Conduct the training loop
    # For either 100 generations or until the first optimizer is done TODO: probably a better way to set this
    print("[INFO] Starting optimization loop...")
    while (gen < 100) and (not ES_LIST[0].stop()):
        # Store the list of solutions (for each controller) from optimizer in a dictionary
        # solutions_dict = {}  # keys are controllers that map to a list of solutions from the optimizer
        solutions_list = []  # indices correspond to controllers
        for es, controller in zip(ES_LIST, CONTROLLER_LIST):
            solutions = es.ask()    # Returns a list of solutions that need to be evaluated (default number is population size)
            solutions_list.append(solutions)    # list of lists of solutions
            # solutions_dict[controller] = solutions
            # print("[DEBUGGING] SHAPE OF SOLUTIONS: {}".format(np.shape(solutions)))
        
        # (slave)/(worker)
        # Create a training environment using the params from the trained RNN model
        env = make_env(args, dream_env=False, render_mode=False, load_model=True)

        # Generate another seed
        seed = seeder.next_seed()
        # print("[DEBUGGING] Seed: {}".format(seed))

        # Generate a list of evaluations from the lists of solutions, f(solutions0, solutions1, solutions2)
        reward_list = []
        t_list = []            
        # print("[DEBUGGING] len of solutions_list: {}".format(len(solutions_list)))
        for wt_idx in range(len(solutions_list[0])):
            # Each index of solutions_list contains a list of solutions (weights) that need to be evaluated
            # TODO: there has to be a better way to do this...
            weights_c0 = solutions_list[0][wt_idx]
            CONTROLLER_LIST[0].set_model_params(weights_c0)
            
            weights_c1 = solutions_list[1][wt_idx]
            CONTROLLER_LIST[1].set_model_params(weights_c1)

            weights_c2 = solutions_list[2][wt_idx]
            CONTROLLER_LIST[2].set_model_params(weights_c2)

            sim_rewards, sim_steps = simulate_multiple_controllers(CONTROLLER_LIST, env,
	        train_mode=1, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
            
            # print("[DEBUGGING] Done evaluating this list of solutions")

            reward_list.append(np.mean(sim_rewards))    # Note we don't have to do this for each controller because it's the same value for all
            t_list.append(np.mean(sim_steps))

        # Check that the result list is equal in size to the solution lists (for each controller)
        for solution in solutions_list:
            assert len(reward_list) == len(solution)

        # (master)
        # Extract some statistics from the simulations
        mean_time_step = int(np.mean(t_list)*100)/100.  # get average time step
        max_time_step = int(np.max(t_list)*100)/100.    # get average time step
        avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
        std_reward = int(np.std(reward_list)*100)/100.  # get average time step

        r_max = int(np.max(reward_list)*100)/100.       # max cummulative reward obtained during evaluation
        r_min = int(np.min(reward_list)*100)/100.       # min cummulative reward obtained during evaluation

        # Get the elapsed time
        curr_time = int(time.time()) - start_time

        # Use the results to update each optimizer and get the current solution for each,
        # set each controller in the global list
        for es, controller in zip(ES_LIST, CONTROLLER_LIST):
            # TODO: I'm not sure if es here is the same thing as ES_LIST[0], i.e. is es a pointer to the ith element of ES_LIST?
            es.tell(reward_list)    # The input to tell needs to be the size as the output from ask()
            
            es_solution = es.result()
            
            model_params = es_solution[0] # best historical solution
            reward = es_solution[1] # best reward, should be the same for each controller
            curr_reward = es_solution[2] # best of the current batch, should be the same for each controller 
            
            controller.set_model_params(np.array(model_params).round(4))

            # Define file names for saving data
            filename = FILEBASE + '.' + str(controller.ID) +  '.json'
            filename_hist = FILEBASE + '.' + str(controller.ID) + '.hist.json'

            # store parameters for each controller
            with open(filename, 'wt') as out:
                res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

            # Store the training history for this optimizer,
            # Note that all the histories are the same for each optimizer except for potentially rms_stdev()
            h = (gen, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)
            history = history_dict[controller]
            history.append(h)   # Update the global variable
            with open(filename_hist, 'wt') as out:
                res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))


        # Every few generations, conduct an evaluation
        if (gen == 1):
            best_reward_eval = avg_reward   # Initialize the best reward on the first pass
        if (gen % eval_steps == 0): # evaluate on actual task at hand
            prev_best_reward_eval = best_reward_eval

            for es, controller in zip(ES_LIST, CONTROLLER_LIST):
                model_params_quantized = np.array(es.current_param()).round(4)
                controller.set_model_params(np.array(model_params).round(4))
            
            # Evaluate using the test_env
            reward_eval_list = evaluate_batch(CONTROLLER_LIST, test_env, test_seed=gen, num_test_episode=num_test_episode, max_len=-1)  # must be updated to support multiple controllers

            # Extract statistics from the evaluation
            reward_eval = np.mean(reward_eval_list)
            r_eval_std = np.std(reward_eval_list)
            r_eval_min = np.min(reward_eval_list)
            r_eval_max = np.max(reward_eval_list)
            e_h = (gen, reward_eval, r_eval_std, r_eval_min, r_eval_max)
            eval_hist.append(e_h)
            with open(filename_eval_hist, 'wt') as out:
                res = json.dump(eval_hist, out, sort_keys=False, indent=0, separators=(',', ':'))

            improvement = reward_eval - best_reward_eval

            for es, controller in zip(ES_LIST, CONTROLLER_LIST):      
                model_params_quantized = np.array(es.current_param()).round(4)
                model_params_list = model_params_quantized.tolist()
                eval_log = eval_dict[controller]    # Get the eval list corresponding to this controller
                eval_log.append([gen, reward_eval, model_params_list])   # Need one of these lists per controller
                # Store the params used for evaluation and the results from it
                filename_log = FILEBASE+ '.' + str(controller.ID) + '.log.json'                   
                with open(filename_log, 'wt') as out:
                    res = json.dump(eval_log, out)  # Will need a file for each controller

                # If this is the best evaluation we've seen, then keep these params for this controller
                if (len(eval_log) == 1 or reward_eval > best_reward_eval):
                    best_reward_eval = reward_eval
                    best_model_params_eval = model_params_list
                else:
                    if retrain_mode:
                        print("reset to previous best params, where best_reward_eval = {}".format(best_reward_eval))
                        es.set_mu(best_model_params_eval)    

                # Store the best parameters for this controller after every evaluation
                filename_best = FILEBASE + '.' + str(controller.ID) + '.best.json'
                with open(filename_best, 'wt') as out:
                    res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))

            # Store the history of best evaluations
            curr_time = int(time.time()) - start_time
            best_record = [gen, curr_time, "improvement", improvement, "current_reward_evaluation", reward_eval, "previous_best_reward", prev_best_reward_eval, "best_reward_so_far", best_reward_eval]
            history_best.append(best_record)
            with open(filename_hist_best, 'wt') as out:
                res = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

        # Increment the generation
        gen += 1
        print("[DEBUGGING] GENERATION COMPLETE")


    print("[INFO] CONTROLLER OPTIMIZATION COMPLETE!")
##########################################################################################################
if __name__ == "__main__":
  args = PARSER.parse_args()
  main(args)