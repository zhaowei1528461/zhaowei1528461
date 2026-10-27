agent_par = {}
agent_par["train"] = True

agent_par["state_dim"] = 53
agent_par["action_dim"] = 4 #0-跟车，1-紧急制动，2-左转，3-右转
agent_par["nb_ego_states"] = 11
agent_par["nb_states_per_vehicle"] = 7
agent_par["sensor_nb_vehicles"] = 6

agent_par["safe_rule"] = False

agent_par["cnn"] = False
agent_par["cnn_state"] = False
if agent_par["cnn"]:
    agent_par["cnn_state"] = True

agent_par["use_quantile"] = True
agent_par["N_QUANT"] = 200
agent_par["use_cvar"] = True
agent_par["cvar_eta"] = 0.7
if agent_par["use_cvar"]:
    agent_par["use_quantile"] = True


agent_par["use_spare_reward"] = False
agent_par["use_her"] = False
agent_par["change_episode"] = 20000
if agent_par["use_spare_reward"]:
    agent_par["use_her"] = True

agent_par["learning_rate"] = 1e-4
agent_par["lr_min"] = 1e-5
agent_par["lr_decay"] = 0.94

agent_par["memory_size"] = 100000
agent_par["Prioritized"] = True
agent_par["prioritized_replay_beta0"] = 0.4
agent_par["prioritized_replay_alpha"] = 0.6

agent_par["N"] = 51
agent_par["Vmin"] = -100
agent_par["Vmax"] = 100
agent_par["seed"] = 914
agent_par["tau"] = 0.01

agent_par["batch_size"] = 32
agent_par["gamma"] = 0.995
agent_par["epsilon"] = 0.8
agent_par["epsilon_min"] = 0.02
agent_par["e_decay"] = 5e5


