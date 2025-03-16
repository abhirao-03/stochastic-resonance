import numpy as np


def PJ_alg(dt: float, noise: np.array):
    delta = int(1000 * (0.01/dt))

    num_steps = int(time_horizon/dt)
    jump_times = np.empty((num_trajectories,))
    x = np.zeros((num_steps,))
    x[0] = x_init
    time_vec = np.linspace(0, time_horizon, num_steps)
    
    for j in tqdm(range(num_trajectories)):
        sign_change_detected = False
        steps_since_change = 0
        initial_sign = np.sign(x[0])
        potential_jump_time = None
        
        for i in range(num_steps - 1):
            curr_t = time_vec[i]
            curr_x = x[i]
            dW = noise[j, i]
            
            x[i+1] = (curr_x + 
                    mu(curr_x, curr_t) * dt + 
                    sigma(curr_x, curr_t) * dW)
            
            # If we haven't detected a sign change yet
            if not sign_change_detected:
                if np.sign(x[i+1]) != initial_sign:
                    sign_change_detected = True
                    steps_since_change = 0
                    potential_jump_time = curr_t
            
            # If we're monitoring a potential zone transition
            elif steps_since_change < delta:
                steps_since_change += 1
                # If sign reverts back to initial sign, cancel the monitoring
                if np.sign(x[i+1]) == initial_sign:
                    sign_change_detected = False
                    steps_since_change = 0
                    potential_jump_time = None
                
                # If we've monitored for delta steps and sign change persists
                if steps_since_change == delta:
                    jump_times[j] = potential_jump_time
                    break
        
        # If we reach the end without confirming a transition
        if not sign_change_detected or steps_since_change < delta:
            jump_times[j] = np.nan
    
    return x, jump_times


def PJDeepWell_V2(dt, noise, jump_threshold=1):
    delta = int((0.0677778 / dt) + 3.22222)
    num_steps = int(time_horizon/dt)
    jump_times = np.empty((num_trajectories,))
    trajectories = np.zeros((num_trajectories, num_steps))
    time_vec = np.linspace(0, time_horizon, num_steps)
    
    for j in tqdm(range(num_trajectories)):
        x = np.zeros((num_steps,))
        x[0] = x_init
        steps_above_threshold = 0
        jump_confirmed = False
        first_cross_index = None
        
        for i in range(num_steps - 1):
            curr_x = x[i]
            dW = noise[j, i]
            
            # Update trajectory
            x[i+1] = curr_x + mu(curr_x, time_vec[i]) * dt + sigma(curr_x, time_vec[i]) * dW
            
            # Check if above jump_threshold
            if x[i+1] >= jump_threshold:
                steps_above_threshold += 1
                if steps_above_threshold == 1:
                    first_cross_index = i + 1  # Index where threshold was first crossed
            else:
                steps_above_threshold = 0
                first_cross_index = None
            
            # Check if persistence criteria met
            if steps_above_threshold >= delta:
                # Backtrack to find last crossing from <0 to >=0 before first_cross_index
                last_zero_cross = None
                if first_cross_index is not None:
                    for k in range(first_cross_index, 0, -1):
                        if x[k] >= 0 and x[k-1] < 0:
                            last_zero_cross = time_vec[k]
                            break
                jump_times[j] = last_zero_cross if last_zero_cross is not None else np.nan
                jump_confirmed = True
                break
        
        if not jump_confirmed:
            jump_times[j] = np.nan
        
        trajectories[j] = x
    
    return trajectories, jump_times