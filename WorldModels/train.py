"""
usage: 
  python train.py -c configs\multiwalker.config


A note on MPI processes:
  It is probably best to think of each MPI process as an independent program, 
  albeit one with the same source code as every other process in the computation.
  Each process has it's own address space. Operations that process 0 carries out
  on variables in its address space have no impact on the contents of the address
  spaces of other processes. In other words, global variables are not shared across
  processes. If we want a slave to be aware of the weights of a controller being
  optimized in another process, we will have to include that information in a packet.
"""

from mpi4py import MPI  # Package for Message Passing Interface, allows python apps to exploit multiple processors 
import numpy as np
import json
import os
import subprocess
import sys
from env import make_env
from controller import make_controller, simulate, simulate_multiple_controllers
from es import CMAES, SimpleGA, OpenES, PEPG
from utils import PARSER
import argparse
import time

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
###

def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
  global population, filebase, game, controller, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE, CONTROLLER_LIST, ES_LIST
  # global population, filebase, game, num_params, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE, CONTROLLER_LIST, ES_LIST
  population = num_worker * num_worker_trial
  filedir = 'results/{}/{}/log/'.format(exp_name, env_name)
  if not os.path.exists(filedir):
      os.makedirs(filedir)
  filebase = filedir+env_name+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  # TODO: Will use 3 files (from series.py) to train each controller

  # If we're using the multiwalker environment, we need to create a controller for each agent
  # TODO: make the number of controllers configurable
  if (env_name == 'multiwalker_v9'):
    controller0 = make_controller(args=config_args, id="Ctrl_0")
    controller1 = make_controller(args=config_args, id="Ctrl_1")
    controller2 = make_controller(args=config_args, id="Ctrl_2")
    CONTROLLER_LIST = [controller0, controller1, controller2]
    # print("[DEBUGGING] Length of controller list: {}".format(len(CONTROLLER_LIST)))
    # print("[DEBUGGING] controller0 ID: {}".format(controller0.ID))

    num_params = controller0.param_count  # all controllers have the same number of dimensions
    print("size of model", num_params)
    print("[DEBUGGING] Optimizer selected: {}".format(optimizer))

    # Instantiate the optimizer to be used for each controller
    # TODO: does an optimizer need to be instantiated for each controller?
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
      cma = CMAES(num_params=num_params,
        sigma_init=sigma_init,
        popsize=population)
      es0 = cma
      es1 = cma
      es2 = cma
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

    ES_LIST = [es0, es1, es2] # TODO: could make this a dictionary that maps controllers to optimizers
    PRECISION = 10000
    SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
    RESULT_PACKET_SIZE = 4*num_worker_trial

  # Else we're using an environment with a single agent
  else:
    controller = make_controller(args=config_args)
    
    CONTROLLER_LIST = [controller]
    
    num_params = controller.param_count
    print("size of model", num_params)

    ## TODO: we will need to instantiate an optimizer for each controller
    if optimizer == 'ses':
      ses = PEPG(num_params,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_alpha=0.2,
        sigma_limit=0.02,
        elite_ratio=0.1,
        weight_decay=0.005,
        popsize=population)
      es = ses
    elif optimizer == 'ga':
      ga = SimpleGA(num_params,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_limit=0.02,
        elite_ratio=0.1,
        weight_decay=0.005,
        popsize=population)
      es = ga
    elif optimizer == 'cma':
      cma = CMAES(num_params,
        sigma_init=sigma_init,
        popsize=population)
      es = cma
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
      es = pepg
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
      es = oes
    ES_LIST = [es]
    PRECISION = 10000
    SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial
    RESULT_PACKET_SIZE = 4*num_worker_trial
###

def sprint(*args):
  print(args) # if python3, can do print(*args)
  sys.stdout.flush()

class OldSeeder:
  def __init__(self, init_seed=0):
    self._seed = init_seed
  def next_seed(self):
    result = self._seed
    self._seed += 1
    return result
  def next_batch(self, batch_size):
    result = np.arange(self._seed, self._seed+batch_size).tolist()
    self._seed += batch_size
    return result

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

def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
  # TODO: do we add information about the specific controller in here?
  n = len(seeds)
  result = []
  worker_num = 0
  for i in range(n):
    worker_num = int(i / num_worker_trial) + 1
    result.append([worker_num, i, seeds[i], train_mode, max_len])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))
  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)
  return result

def decode_solution_packet(packet):
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float)/PRECISION])
  return result

def encode_result_packet(results):
  r = np.array(results)
  r[:, 2:4] *= PRECISION
  return r.flatten().astype(np.int32)

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, 4)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  fits = r[:, 2].astype(np.float)/PRECISION
  fits = fits.tolist()
  times = r[:, 3].astype(np.float)/PRECISION
  times = times.tolist()
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], fits[i], times[i]])
  return result

def worker(weights, seed, train_mode_int=1, max_len=-1):

  print("[DEBUGGING] Worker is working")
  print("[DEBUGGING] Length of CONTROLLER_LIST: {}".format(len(CONTROLLER_LIST)))
  train_mode = (train_mode_int == 1)
  # TODO: We should update the weights for all the controllers here
  # for c, wts in zip(CONTROLLER_LIST, weights_list): c.set_model_weights(wts)
  # if controller_id == 0:
  #   controller.set_model_params(weights)
  # elif controller_id == 1:
  #   CONTROLLER_LIST[1].set_model_params(weights)
  # ...
  # Using the above paradigm, I don't think that the weights of the other controllers will be set. 

  controller.set_model_params(weights)

  if env_name=='multiwalker_v9':
    if train_mode_int == True:
      reward_list, t_list = simulate_multiple_controllers(CONTROLLER_LIST, env,
        train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
    else:
      reward_list, t_list = simulate_multiple_controllers(CONTROLLER_LIST, test_env,
          train_mode=train_mode, render_mode=False, num_episode=num_test_episode, seed=seed, max_len=max_len)
  # Else we're using car-racing or doom env
  else:
    if train_mode_int == True:
      reward_list, t_list = simulate(controller, env,
        train_mode=train_mode, render_mode=False, num_episode=num_episode, seed=seed, max_len=max_len)
    else:
      reward_list, t_list = simulate(controller, test_env,
          train_mode=train_mode, render_mode=False, num_episode=num_test_episode, seed=seed, max_len=max_len)
  
  if batch_mode == 'min':
    reward = np.min(reward_list)
  else: ## batch_mode == 'mean'
    reward = np.mean(reward_list) ## mean total reward from all simulations
  t = np.mean(t_list)             ## mean number of steps during all simulations
  # print(t, reward)
  print("[DEBUGGING] Results from simulating controller: mean number of steps taken = {}, mean total reward = {}".format(t,reward))
  return reward, t

def slave():
  global env ## TODO: deconflict name with what's used in mpi_fork()?
  # Create an env to conduct training, note that here we use the rnn params trained in rnn_train.py
  if env_name == 'CarRacing-v0' or env_name == 'multiwalker_v9':
    print("[DEBUGGING] Creating env for slave process...")
    env = make_env(args=config_args, dream_env=False, load_model=True) # training in dreams not supported yet
  else:
    env = make_env(args=config_args, dream_env=True, render_mode=False, load_model=True)

  packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
  while 1:  # TODO: how does this loop end?
    comm.Recv(packet, source=0)
    # print("[DEBUGGING] packet received for controller: {}".format(controller.ID))
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    solutions = decode_solution_packet(packet)  ## solutions is specific to a single controller
    results = []
    for solution in solutions:
      worker_id, jobidx, seed, train_mode, max_len, weights = solution
      assert (train_mode == 1 or train_mode == 0), str(train_mode)
      worker_id = int(worker_id)
      possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
      assert worker_id == rank, possible_error
      jobidx = int(jobidx)
      seed = int(seed)
      fitness, timesteps = worker(weights, seed, train_mode, max_len) ## Here we need to call worker for each controller
      results.append([worker_id, jobidx, fitness, timesteps])
    result_packet = encode_result_packet(results)
    assert len(result_packet) == RESULT_PACKET_SIZE
    comm.Send(result_packet, dest=0)

def send_packets_to_slaves(packet_list):
  num_worker = comm.Get_size()
  assert len(packet_list) == num_worker-1
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    assert(len(packet) == SOLUTION_PACKET_SIZE)
    comm.Send(packet, dest=i)

def receive_packets_from_slaves():
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

  reward_list_total = np.zeros((population, 2))

  check_results = np.ones(population, dtype=np.int)
  for i in range(1, num_worker+1):
    comm.Recv(result_packet, source=i)
    results = decode_result_packet(result_packet)
    for result in results:
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, 0] = result[2]
      reward_list_total[idx, 1] = result[3]
      check_results[idx] = 0

  check_sum = check_results.sum()
  assert check_sum == 0, check_sum
  return reward_list_total

def evaluate_batch(model_params, test_seed, max_len=-1):
  # runs only from master since mpi and Doom was janky
  controller.set_model_params(model_params)
  ## TODO: will need to modify simulate() to use multiple controllers
  rewards_list, t_list = simulate(controller, test_env,
        train_mode=False, render_mode=False, num_episode=num_test_episode, seed=test_seed, max_len=max_len) ## TODO: why is train_mode false here?
  return rewards_list

def master():
  # global test_env, es, controller # TODO: Get rid of global variables
  global test_env
  # Construct the environment to be used to evaluate the controller at various stages in the optimization process
  # This env is used when train_mode is set to False in simulate(...)
  # Note that the env here is created using the rnn params that were trained in rnn_train.py
  print("[DEBUGGING] Creating env for master process...")
  if env_name == 'CarRacing-v0':
    test_env = make_env(args=config_args, dream_env=False, load_model=True)
  else:
    # Use these environment arguments for Doom and Multiwalker
    test_env = make_env(args=config_args, dream_env=False, render_mode=False, load_model=True)


  # for c in CONTROLLER_LIST:
  #   print("[DEBUGGING] Ctrl ID: {}".format(c.ID))


  start_time = int(time.time())
  sprint("training", env_name)
  #sprint("population", es.popsize)
  sprint("num_worker", num_worker)
  sprint("num_worker_trial", num_worker_trial)
  sys.stdout.flush()

  seeder = Seeder(seed_start)

  # # We have 3 controllers so may need to create 3 sets of files
  # filename = filebase+'.json'
  # filename_log = filebase+'.log.json'
  # filename_hist = filebase+'.hist.json'
  # filename_eval_hist = filebase+'.eval_hist.json'
  # filename_hist_best = filebase+'.hist_best.json'
  # filename_best = filebase+'.best.json'
  
  t = 0  # "Generation (for all optimizers)"

  history = []
  history_best = [] # stores evaluation averages every 25 steps or so
  eval_log = []
  eval_hist = []
  best_reward_eval = 0
  best_model_params_eval = None

  max_len = -1 # max time steps (-1 means ignore)
  while True: # TODO: how does this loop end? should this be "while not es.stop()" 
    
    # loop through all optimzers
    # for es, controller in zip(ES_LIST, CONTROLLER_LIST):
    # ask/tell interface is one way of running the optimizer
    # ask returns new candidate solutions, sampled from a multi-variate normal distribution and transformed to f-representation (phenotype) to be evaluated.	 
    for e, ctrl in zip(ES_LIST, CONTROLLER_LIST):
      
      es = e
      controller = ctrl
      
      print("[DEBUGGING] Generation: {} Controller: {}".format(t, controller.ID))
    
      # We have 3 controllers so may need to create 3 sets of files
      filename = filebase+'_'+controller.ID+'.json'
      filename_log = filebase+'_'+controller.ID+'.log.json'
      filename_hist = filebase+'_'+controller.ID+'.hist.json'
      filename_eval_hist = filebase+'_'+controller.ID+'.eval_hist.json'
      filename_hist_best = filebase+'_'+controller.ID+'.hist_best.json'
      filename_best = filebase+'_'+controller.ID+'.best.json'
    
      solutions = es.ask()  # solutions for this controller/optimizer pair

      if antithetic:
        seeds = seeder.next_batch(int(es.popsize/2))
        seeds = seeds+seeds
      else:
        seeds = seeder.next_batch(es.popsize)
      packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

      send_packets_to_slaves(packet_list)
      reward_list_total = receive_packets_from_slaves()

      reward_list = reward_list_total[:, 0] # get rewards

      mean_time_step = int(np.mean(reward_list_total[:, 1])*100)/100. # get average time step
      max_time_step = int(np.max(reward_list_total[:, 1])*100)/100. # get average time step
      avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
      std_reward = int(np.std(reward_list)*100)/100. # get average time step
      
      # Tell updates the optimizer instance by passing respective function values
      es.tell(reward_list)

      es_solution = es.result() # Returns (xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xmean), effective_stds)
      model_params = es_solution[0] # best historical solution
      reward = es_solution[1] # best reward
      curr_reward = es_solution[2] # best of the current batch
      controller.set_model_params(np.array(model_params).round(4))

      r_max = int(np.max(reward_list)*100)/100. 
      r_min = int(np.min(reward_list)*100)/100.

      curr_time = int(time.time()) - start_time

      h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

      if cap_time_mode:
        max_len = 2*int(mean_time_step+1.0)
      else:
        max_len = -1

      history.append(h)

      with open(filename, 'wt') as out:
        res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

      with open(filename_hist, 'wt') as out:
        res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

      # sprint(env_name, h)
      print("[INFO] env_name: {}, h: {}".format(env_name, h))
      
      if (t == 1):
        best_reward_eval = avg_reward
      if (t % eval_steps == 0): # evaluate on actual task at hand

        prev_best_reward_eval = best_reward_eval
        model_params_quantized = np.array(es.current_param()).round(4)
        reward_eval_list = evaluate_batch(model_params_quantized, max_len=-1, test_seed=t)
        reward_eval = np.mean(reward_eval_list)
        r_eval_std = np.std(reward_eval_list)
        r_eval_min = np.min(reward_eval_list)
        r_eval_max = np.max(reward_eval_list)
        model_params_quantized = model_params_quantized.tolist()
        improvement = reward_eval - best_reward_eval
        eval_log.append([t, reward_eval, model_params_quantized])
        e_h = (t, reward_eval, r_eval_std, r_eval_min, r_eval_max)
        eval_hist.append(e_h)
        with open(filename_eval_hist, 'wt') as out:
          res = json.dump(eval_hist, out, sort_keys=False, indent=0, separators=(',', ':'))
        with open(filename_log, 'wt') as out:
          res = json.dump(eval_log, out)
        if (len(eval_log) == 1 or reward_eval > best_reward_eval):
          # New reward is the best we've seen so far so store this value 
          best_reward_eval = reward_eval 
          # Store the params used to generate the best reward 
          best_model_params_eval = model_params_quantized
        else:
          if retrain_mode:
            sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
            es.set_mu(best_model_params_eval) # Only implemented for OpenES and PEPG, does not do anything for CMAES
        with open(filename_best, 'wt') as out:
          res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
        # dump history of best
        curr_time = int(time.time()) - start_time
        best_record = [t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval]
        history_best.append(best_record)
        with open(filename_hist_best, 'wt') as out:
          res = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

        sprint("Eval", t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


    # increment generation once all controllers have completed the optimization loop
    t += 1


def main(args):
  # TODO: get rid of global variables...
  global optimizer, num_episode, num_test_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode, env_name, exp_name, batch_mode, config_args

  optimizer = args.controller_optimizer
  num_episode = args.controller_num_episode
  num_test_episode = args.controller_num_test_episode
  eval_steps = args.controller_eval_steps
  num_worker = args.controller_num_worker ## May be a good idea to keep this very small for running on a personal laptop
  num_worker_trial = args.controller_num_worker_trial
  antithetic = (args.controller_antithetic == 1)
  retrain_mode = (args.controller_retrain == 1)
  cap_time_mode= (args.controller_cap_time == 1)
  seed_start = args.controller_seed_start
  env_name = args.env_name
  exp_name = args.exp_name
  batch_mode = args.controller_batch_mode
  config_args = args

  initialize_settings(args.controller_sigma_init, args.controller_sigma_decay)
  
  print("[DEBUGGING] SUCCESFULY CALLED initialize_settings")

  sprint("process", rank, "out of total ", comm.Get_size(), "started")
  if (rank == 0):
    master()
    print("[DEBUGGING] SUCCESSFULLY CALLED master")
  else:
    slave() # TODO: do we need to create a slave() for each controller?
    print("[DEBUGGING] SUCCESSFULLY CALLED slave")

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    try:
      #print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
      #subprocess.check_call(["mpirun", "--allow-run-as-root", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env) # This is an OpenMPI call
      print( ["mpiexec", "-n", str(n), sys.executable] + sys.argv)
      subprocess.check_call(["mpiexec", "-n", str(n), sys.executable] + sys.argv, env=env)  # MSMPI equivalent command
      return "parent"
    except subprocess.CalledProcessError as e:
      print("[ERROR] {}".format(e))
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank: {} and nworkers: {}'.format(rank, nworkers))
    return "child"

if __name__ == "__main__":
  args = PARSER.parse_args()
  if "parent" == mpi_fork(args.controller_num_worker+1): os.exit()  ## TODO: make this configurable so we don't have to use threading?
  
  print("[DEBUGGING] SUCCESFULY CALLED mpi_fork")
  
  main(args)
