import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import make_interp_spline
from sklearn.isotonic import IsotonicRegression

def get_next_batch_to_run(x_obs, y_obs, n_parallel, x_range=(-5, 5)):
    """
    Selects N different points to run on your GPUs simultaneously.
    Uses 'Virtual Updates' only to ensure the N points are spread out.
    """
    # 1. Fit the GP on your REAL data
    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(x_obs, y_obs)

    x_dense = np.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
    
    current_suggestions = []
    
    # We create a 'temporary' GP for this batch selection session
    tmp_x = x_obs.copy()
    tmp_y = y_obs.copy()

    for _ in range(n_parallel):
        # Predict uncertainty
        _, sigma = gp.predict(x_dense, return_std=True)
        
        # Pick the point where uncertainty is highest
        best_idx = np.argmax(sigma)
        next_x = x_dense[best_idx]
        current_suggestions.append(next_x)
        
        # VIRTUAL UPDATE:
        # We tell the GP: "Pretend we know the answer here" 
        # so GPU #2 finds a different spot.
        y_virt = gp.predict(next_x.reshape(-1, 1))
        tmp_x = np.vstack([tmp_x, next_x])
        tmp_y = np.append(tmp_y, y_virt)
        
        # Refit the GP with the virtual point
        gp.fit(tmp_x, tmp_y)

    return np.array(current_suggestions).flatten(), np.max(sigma)

def suggest_batch_by_y_spread(x_obs, y_obs, n_samples=11, x_range=(-5, 5), banned_x=None):
    if len(x_obs) < 2:
        return np.array([])

    # 1. Sort data (PCHIP requires strictly increasing x)
    idx = np.argsort(x_obs)
    xs, ys = x_obs[idx], y_obs[idx]
    
    # 2. Create a Monotonic Spline 
    interp = PchipInterpolator(xs, ys)
    
    # 3. Create a dense 'search' grid
    x_dense = np.linspace(x_range[0], x_range[1], 1000)
    y_dense = interp(x_dense)
    
    # 4. Define our target Y-grid (Evenly spaced)
    y_targets = np.linspace(np.min(ys), np.max(ys), n_samples)
    
    suggested_x = []
    selected_xs = []
    for target in y_targets:
        ranked_indices = np.argsort(np.abs(y_dense - target))
        for candidate_idx in ranked_indices:
            target_x = x_dense[candidate_idx]
            min_obs_dist = np.min(np.abs(x_obs - target_x))
            if min_obs_dist <= 0.1:
                continue
            if banned_x is not None and len(banned_x) > 0:
                if np.min(np.abs(banned_x - target_x)) <= 0.1:
                    continue
            if selected_xs and np.min(np.abs(np.array(selected_xs) - target_x)) <= 0.1:
                continue
            suggested_x.append((target_x, min_obs_dist))
            selected_xs.append(target_x)
            break
            
    # 5. Sort suggestions by how "isolated" they are (largest min_dist first)
    suggested_x.sort(key=lambda val: val[1], reverse=True)
    final_batch = [x[0] for x in suggested_x[:n_samples]]
    
    return np.array(final_batch)

if __name__ == "__main__":
    # --- YOUR MAIN LOOP ---

    # # Start with a simple range
    # x_obs = np.array([-5, 0, 5]).reshape(-1, 1)
    # # You run your GenAI model here to get initial Ys
    # y_obs = np.array([0.0, 0.4, 0.9]) 

    # N_GPUS = 4

    # for round in range(5): # Run for 5 minutes total
    #     # 1. Get the best X values for our GPUs
    #     next_x_batch, uncertainty = get_next_batch_to_run(x_obs, y_obs, N_GPUS)
        
    #     print(f"Round {round} | Max Uncertainty: {uncertainty:.4f}")
    #     print(f"Sending to GPUs: {next_x_batch}")

    #     # 2. RUN YOUR GenAI MODEL (This is your 1-minute wait)
    #     # y_new = run_genai_on_gpus(next_x_batch) 
    #     y_new = np.random.random(N_GPUS) # Placeholder for real AI results

    #     # 3. Update our REAL observation list
    #     x_obs = np.vstack([x_obs, next_x_batch.reshape(-1, 1)])
    #     y_obs = np.append(y_obs, y_new)
        
    #     if uncertainty < 0.05: # Stop if we are confident enough
    #         break
    example_data = np.array([[-5.00000,-0.03718],
                            [-3.75000,-0.03095],
                            [-3.28829,-0.03017],
                            [-3.00801,-0.02775],
                            [-2.91792,-0.02282],
                            [-2.50000,-0.01883],
                            [-1.25000,-0.02089],
                            [-0.67568,-0.01361],
                            [-0.55556,-0.01533],
                            [-0.35536,-0.01057],
                            [-0.16517,-0.00628],
                            [0.00000,0.00000],
                            [0.22523,0.00635],
                            [1.37638,0.02117],
                            [1.75676,0.06316],
                            [2.50000,0.03478],
                            [2.94795,0.03666],
                            [3.07808,0.03927],
                            [3.75000,0.04446],
                            [4.77978,0.04889],
                            [5.00000,0.05173]])
    
    x_samples = example_data[:, 0]
    y_samples = example_data[:, 1]

    # 2. FORCE MONOTONICITY (Isotonic Regression)
    # This removes the dips by finding the best-fit non-decreasing sequence.
    ir = IsotonicRegression(out_of_bounds='clip')
    y_corrected = ir.fit_transform(x_samples, y_samples)

    # 3. SMOOTH THE RESULT (B-Spline)
    # Now that the dips are gone, we can draw a smooth line through the corrected points.
    x_smooth = np.linspace(x_samples.min(), x_samples.max(), 300)
    spline = make_interp_spline(x_samples, y_corrected, k=3)
    y_smooth = spline(x_smooth)

    # Plotting to show you the difference
    plt.scatter(x_samples, y_samples, color='blue', label='Noisy Observations')
    plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label='Monotonic Regressor (Smooth)')
    plt.legend()
    plt.xlabel('Slider Value')
    plt.ylabel('LPIPS Score')
    plt.savefig('interpolated_curve_isotonic_regression.png')
    # plt.show()

    # interp = PchipInterpolator(x_samples, y_samples)
    
    # # 3. Create a dense 'search' grid
    # x_dense = np.linspace(np.min(x_samples), np.max(x_samples), 1000)
    # y_dense = interp(x_dense)
    # # plot observations and interpolated curve
    # plt.plot(x_samples, y_samples, 'o', label='Observations', color='blue')
    # plt.plot(x_dense, y_dense, label='Interpolated Curve', color='orange')
    # plt.legend()
    # plt.xlabel('Slider Value')
    # plt.ylabel('LPIPS Score')
    # plt.savefig('interpolated_curve.png')
    # plt.close()

    # next_gpu_batch = suggest_batch_by_y_spread(x_samples, y_samples)
    # print(f"Next X values to sample: {next_gpu_batch}")


'''
def get_lpips_curve_gaussian_process(
    view_images_dir: str,
    edited_images_dir: str,
    boundaries: Tuple[float, float],
    out_dir: str = None,
    gpu_ids: List[int] = [0],
    n_samples: int = 11,
    workers_per_gpu: int = 1,
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
):
    lpips_model = get_lpips_model(f'cuda:{gpu_ids[0]}')
    if pool is None:
        if pipelines is None:
            pipelines = load_pipelines(gpu_ids)
        if conds is None:
            conds = load_conds(pipelines, view_images_dir, edited_images_dir)
    pos_boundary, neg_boundary = boundaries
    ori_slider_values = np.linspace(neg_boundary, pos_boundary, n_samples).tolist()
    # initial observations
    values_to_generate = [
        sv for sv in ori_slider_values
        if not os.path.exists(os.path.join(out_dir, f"search_at_{sv:.5f}"))
        or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{sv:.5f}")))
    ]
    time_to_generate_initial_frames = time.time()
    T0 = time.time()
    # step 1. generate key-frames for initial observations if not generated already
    if values_to_generate:
        run_generation_multiprocess(
            view_images_dir, edited_images_dir, values_to_generate, out_dir, gpu_ids, workers_per_gpu,
            pipelines=pipelines, pool=pool, conds=conds)
    time_to_generate_initial_frames = time.time() - time_to_generate_initial_frames
    # step 2. compute LPIPS from saved frames
    lpips_scores = get_LPIPS_score_relative_to_reference(lpips_model, neg_boundary, ori_slider_values, out_dir)

    max_uncertainty = 1
    observation_slider_values = np.array(ori_slider_values).reshape(-1, 1)
    observation_lpips_scores = np.array(lpips_scores)
    round = 0
    time_to_generate_new_frames = time.time()
    while max_uncertainty > 0.001:
        # step 3. get next batch to run
        print('searching for next batch to run...')
        next_sv_batch, max_uncertainty = get_next_batch_to_run(observation_slider_values, observation_lpips_scores, len(gpu_ids), x_range=(neg_boundary, pos_boundary))
        # remove svs that are already in observation_slider_values
        next_sv_batch = [sv for sv in next_sv_batch if sv not in observation_slider_values.flatten()]
        print(f"Round {round+1} | Max Uncertainty: {max_uncertainty:.4f}")
        print(f"Next slider values to generate: {next_sv_batch}")
        if not next_sv_batch:
            print('No new slider values to generate. Stopping...')
            break
        new_svs_to_generate = [
            sv for sv in next_sv_batch
            if not os.path.exists(os.path.join(out_dir, f"search_at_{sv:.5f}"))
            or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{sv:.5f}")))
        ]
        if new_svs_to_generate:
            run_generation_multiprocess(
                view_images_dir, edited_images_dir, new_svs_to_generate, out_dir, gpu_ids, 1,
                pipelines=pipelines, pool=pool, conds=conds)
        new_lpips_scores = get_LPIPS_score_relative_to_reference(lpips_model, neg_boundary, next_sv_batch, out_dir)
        observation_slider_values = np.vstack([observation_slider_values, np.array(next_sv_batch).reshape(-1, 1)])
        observation_lpips_scores = np.append(observation_lpips_scores, new_lpips_scores)
        round += 1
    time_to_generate_new_frames = time.time() - time_to_generate_new_frames
    T1 = time.time()
    print(f"Time taken: {T1 - T0:.2f} seconds")
    # sort observation_slider_values and observation_lpips_scores by observation_slider_values
    sorted_indices = np.argsort(observation_slider_values.flatten())
    observation_slider_values = observation_slider_values[sorted_indices]
    observation_lpips_scores = observation_lpips_scores[sorted_indices]
    with open(os.path.join(out_dir, "lpips_curve.txt"), "w") as f:
        for ori_sv, ls in zip(observation_slider_values.flatten(), observation_lpips_scores):
            f.write(f"{ori_sv:.5f},{ls:.5f}\n")
        f.write("\n")
        f.write(f"Time taken to generate initial frames: {time_to_generate_initial_frames:.2f} seconds\n")
        f.write(f"Time taken to generate new frames: {time_to_generate_new_frames:.2f} seconds\n")
        f.write(f"Total time taken: {T1 - T0:.2f} seconds\n")
        f.write(f"Number of rounds: {round}\n")
        f.write(f"Final max uncertainty: {max_uncertainty:.5f}\n")

    
    # plot the line connecting points corresponding to lowest and highest lpips scores
    lowest_lpips_score_index = np.argmin(observation_lpips_scores)
    highest_lpips_score_index = np.argmax(observation_lpips_scores)
    plt.plot([observation_slider_values.flatten()[lowest_lpips_score_index], observation_slider_values.flatten()[highest_lpips_score_index]], 
        [observation_lpips_scores[lowest_lpips_score_index], observation_lpips_scores[highest_lpips_score_index]], '--', color='red')
    # plot the lpips curve
    plt.plot(observation_slider_values.flatten(), observation_lpips_scores, '-o', markersize=5)
    plt.xlabel('Slider Value')
    plt.ylabel('LPIPS Score')
    plt.savefig(os.path.join(out_dir, "lpips_curve.png"))
    plt.close()

    return observation_slider_values, observation_lpips_scores
'''