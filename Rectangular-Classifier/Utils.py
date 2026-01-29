import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(42)
N_POOL = 100000      # Size of the unlabeled pool
N_ITERATIONS = 5     # Number of labeling rounds
N_EXPERIMENTS = 30     # Number of repeated experiments to average
TARGET_RECT = [0.2, 0.8, 0.2, 0.8]  # [x1, x2, y1, y2] True rectangle range

# ==========================================
# 1.Utils
# ==========================================
def get_label(points):
    """Determine whether the point is within the actual rectangle"""
    x, y = points[:, 0], points[:, 1]
    return ((x >= TARGET_RECT[0]) & (x <= TARGET_RECT[1]) & 
            (y >= TARGET_RECT[2]) & (y <= TARGET_RECT[3])).astype(int)

# ==========================================
# 2.Classifier: RectangleModel
# ==========================================
class RectangleModel:
    def __init__(self):
        self.rect = None # [x_min, x_max, y_min, y_max]
        
    def fit(self, X, y):
        pos_samples = X[y == 1]
        if len(pos_samples) > 0:
            self.rect = [
                pos_samples[:, 0].min(), pos_samples[:, 0].max(),
                pos_samples[:, 1].min(), pos_samples[:, 1].max()
            ]
        else:
            self.rect = None

    def predict(self, X):
        if self.rect is None: return np.zeros(len(X))
        x, y = X[:, 0], X[:, 1]
        return ((x >= self.rect[0]) & (x <= self.rect[1]) & 
                (y >= self.rect[2]) & (y <= self.rect[3])).astype(int)

def calculate_iou(rect1, rect2):
    """Calculate the Intersection over Union (IoU) of two rectangles as an evaluation metric"""
    if rect1 is None or rect2 is None: return 0
    curr_x1, curr_x2, curr_y1, curr_y2 = rect1
    true_x1, true_x2, true_y1, true_y2 = rect2
    
    inter_x1 = max(curr_x1, true_x1)
    inter_x2 = min(curr_x2, true_x2)
    inter_y1 = max(curr_y1, true_y1)
    inter_y2 = min(curr_y2, true_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
    area2 = (true_x2 - true_x1) * (true_y2 - true_y1)
    return inter_area / (area1 + area2 - inter_area)

def run_experiment(strategy='random', k=1):
    # Initialize data pool
    X_pool = np.random.rand(N_POOL, 2)
    y_pool = get_label(X_pool)
    
    # Initial seeds: try to select 1 positive sample + 1 negative sample; if not possible, randomly select 2 points
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]

    if len(pos_idx) > 0 and len(neg_idx) > 0:
        idx_pos = np.random.choice(pos_idx, 1)
        idx_neg = np.random.choice(neg_idx, 1)
        idx = np.concatenate([idx_pos, idx_neg])
    else:
        # If the pool contains all positive or all negative samples, degrade to randomly selecting 2 different points
        n_init = min(2, len(y_pool))
        idx = np.random.choice(len(y_pool), n_init, replace=False)
    
    X_train = X_pool[idx]
    y_train = y_pool[idx]
    X_pool = np.delete(X_pool, idx, axis=0)
    y_pool = np.delete(y_pool, idx, axis=0)
    
    history = []
    model = RectangleModel()

    # First fit with the initial two points and record the performance at "iteration 0"
    model.fit(X_train, y_train)
    history.append(calculate_iou(model.rect, TARGET_RECT))

    # Then perform N_ITERATIONS rounds, each round adding k points and recording the performance at the end of the round
    for i in range(N_ITERATIONS):
        k_curr = min(k, len(X_pool))  # Prevent pool from running out
        if k_curr == 0:
            # If the pool is exhausted, keep the current IoU for subsequent rounds (extend horizontally)
            history.append(history[-1])
            continue

        if strategy == 'random':
            sel_idx = np.random.choice(len(X_pool), size=k_curr, replace=False)
        else:  # active
            if model.rect is None:
                sel_idx = np.random.choice(len(X_pool), size=k_curr, replace=False)
            else:
                d_x1 = np.abs(X_pool[:, 0] - model.rect[0])
                d_x2 = np.abs(X_pool[:, 0] - model.rect[1])
                d_y1 = np.abs(X_pool[:, 1] - model.rect[2])
                d_y2 = np.abs(X_pool[:, 1] - model.rect[3])
                dist = np.minimum.reduce([d_x1, d_x2, d_y1, d_y2])
                sel_idx = np.argsort(dist)[:k_curr]   # Select k points closest to the rectangle edges

        # Update training set (add k points at once)
        X_train = np.vstack([X_train, X_pool[sel_idx]])
        y_train = np.concatenate([y_train, y_pool[sel_idx]])
        X_pool = np.delete(X_pool, sel_idx, axis=0)
        y_pool = np.delete(y_pool, sel_idx, axis=0)

        model.fit(X_train, y_train)
        history.append(calculate_iou(model.rect, TARGET_RECT))

    return history



def run_experiment_with_config(N_pool, n_iterations, strategy='random', k=None):
    """Run multiple experiments with given N_pool and n_iterations configurations and return the result matrix.

    The return shape is (N_EXPERIMENTS, n_iterations + 1).
    """
    global N_POOL, N_ITERATIONS

    old_pool, old_iter = N_POOL, N_ITERATIONS
    try:
        N_POOL, N_ITERATIONS = N_pool, n_iterations
        # If k is not specified, default to evenly distributing the labeling budget based on the current configuration
        k_case = k if k is not None else max(1, N_POOL // N_ITERATIONS)
        results = np.array([
            run_experiment(strategy=strategy, k=k_case)
            for _ in range(N_EXPERIMENTS)
        ])
    finally:
        N_POOL, N_ITERATIONS = old_pool, old_iter

    return results


def sweep_and_plot(N_pool_list=None, N_iterations_list=None, strategy='active', k=None):
    """Plot IoU curves under different N_POOL or N_ITERATIONS settings.
    Only one of N_pool_list or N_iterations_list should be provided.
    """
    if (N_pool_list is None and N_iterations_list is None) or \
       (N_pool_list is not None and N_iterations_list is not None):
        raise ValueError("Please specify only one of N_pool_list or N_iterations_list")

    plt.figure(figsize=(10, 6))

    # Case 1: Fix N_ITERATIONS, sweep different N_POOL values
    if N_pool_list is not None:
        for Np in N_pool_list:
            results = run_experiment_with_config(Np, N_ITERATIONS, strategy=strategy, k=k)
            mean_curve = results.mean(axis=0)
            std_curve = results.std(axis=0)
            x = np.arange(N_ITERATIONS + 1)
            label = f"{strategy}, N_POOL={Np}"
            plt.plot(x, mean_curve, lw=2, label=label)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.1)

    # Case 2: Fix N_POOL, sweep different N_ITERATIONS values
    if N_iterations_list is not None:
        for Ni in N_iterations_list:
            results = run_experiment_with_config(N_POOL, Ni, strategy=strategy, k=k)
            mean_curve = results.mean(axis=0)
            std_curve = results.std(axis=0)
            x = np.arange(Ni + 1)
            label = f"{strategy}, N_ITER={Ni}"
            plt.plot(x, mean_curve, lw=2, label=label)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.1)

    plt.title("IoU vs Iterations under Different Settings")
    plt.xlabel("Iteration")
    plt.ylabel("IoU")
    # The x-axis starts from 0, where 0 represents the performance with only the initial seed
    if N_pool_list is not None:
        plt.xticks(np.arange(N_ITERATIONS + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_random_vs_active_by_pool(N_pool_list, k=None):
    """Given a list of N_POOL values, plot the random/active curves for different N_POOLs in multiple subplots of a single figure."""
    n_cases = len(N_pool_list)
    n_cols = min(4, n_cases)
    n_rows = int(np.ceil(n_cases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, Np in zip(axes, N_pool_list):
        results_random = run_experiment_with_config(Np, N_ITERATIONS, strategy='random', k=k)
        results_active = run_experiment_with_config(Np, N_ITERATIONS, strategy='active', k=k)

        mean_r = results_random.mean(axis=0)
        std_r = results_random.std(axis=0)
        mean_a = results_active.mean(axis=0)
        std_a = results_active.std(axis=0)

        x = np.arange(N_ITERATIONS + 1)

        ax.plot(x, mean_r, label='Random', color='blue', lw=2)
        ax.plot(x, mean_a, label='Active', color='red', lw=2)
        ax.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.1, color='blue')
        ax.fill_between(x, mean_a - std_a, mean_a + std_a, alpha=0.1, color='red')

        ax.set_title(f"N_POOL={Np}")
        ax.set_xlabel("Iteration")
        # Only keep integer ticks on the x-axis
        ax.set_xticks(np.arange(N_ITERATIONS + 1))

    # Hide extra empty subplots
    for ax in axes[n_cases:]:
        ax.axis('off')

    axes[0].set_ylabel("IoU")
    axes[0].legend(loc='lower right')
    fig.suptitle(f"Random vs Active under different N_POOL (N_ITER={N_ITERATIONS})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Example: Plot by different iteration counts, each plot contains random and active
def plot_random_vs_active_by_iterations(N_iterations_list, k=None):
    """Given a list of N_ITERATIONS, plot the random/active curves for different N_ITER in multiple subplots of a single figure."""

    n_cases = len(N_iterations_list)
    n_cols = min(4, n_cases)
    n_rows = int(np.ceil(n_cases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, Ni in zip(axes, N_iterations_list):
        results_random = run_experiment_with_config(N_POOL, Ni, strategy='random', k=k)
        results_active = run_experiment_with_config(N_POOL, Ni, strategy='active', k=k)

        mean_r = results_random.mean(axis=0)
        std_r = results_random.std(axis=0)
        mean_a = results_active.mean(axis=0)
        std_a = results_active.std(axis=0)

        x = np.arange(Ni + 1)

        ax.plot(x, mean_r, label='Random', color='blue', lw=2)
        ax.plot(x, mean_a, label='Active', color='red', lw=2)
        ax.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.1, color='blue')
        ax.fill_between(x, mean_a - std_a, mean_a + std_a, alpha=0.1, color='red')

        ax.set_title(f"N_ITER={Ni}")
        ax.set_xlabel("Iteration")
        # Only keep integer ticks on the x-axis
        ax.set_xticks(np.arange(Ni + 1))

    # Hide extra empty subplots
    for ax in axes[n_cases:]:
        ax.axis('off')

    axes[0].set_ylabel("IoU")
    axes[0].legend(loc='lower right')
    fig.suptitle(f"Random vs Active under different N_ITER (N_POOL={N_POOL})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_random_vs_active_by_k(k_list, total_budget,n_iterations=None):
    """Plot learning curves for different batch sizes k under a fixed total annotation budget.

    For each k, calculate the maximum number of iterations n_iter_k ≈ total_budget / k,
    then plot the average IoU curves for random and active strategies under that k in a single subplot.
    """

    n_cases = len(k_list)
    n_cols = min(4, n_cases)
    n_rows = int(np.ceil(n_cases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, k_val in zip(axes, k_list):
        # Given total_budget and k, approximate the number of iterations (ignoring the initial 2 seed points)
        # n_iter_k = max(1, total_budget // k_val)
        n_iter_k = n_iterations if n_iterations is not None else max(1, total_budget // k_val)

        results_random = run_experiment_with_config(total_budget, n_iter_k,
                                                    strategy='random', k=k_val)
        results_active = run_experiment_with_config(total_budget, n_iter_k,
                                                    strategy='active', k=k_val)

        mean_r = results_random.mean(axis=0)
        std_r = results_random.std(axis=0)
        mean_a = results_active.mean(axis=0)
        std_a = results_active.std(axis=0)

        x = np.arange(n_iter_k + 1)

        ax.plot(x, mean_r, label='Random', color='blue', lw=2)
        ax.plot(x, mean_a, label='Active', color='red', lw=2)
        ax.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.1, color='blue')
        ax.fill_between(x, mean_a - std_a, mean_a + std_a, alpha=0.1, color='red')

        ax.set_title(f"budget/round={k_val}, rounds={n_iter_k}")
        ax.set_xlabel("Iteration")
        ax.set_xticks(np.arange(n_iter_k + 1))

    # Hide extra empty subplots
    for ax in axes[n_cases:]:
        ax.axis('off')

    axes[0].set_ylabel("IoU")
    axes[0].legend(loc='lower right')
    fig.suptitle(f"Random vs Active under fixed budget={total_budget}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_random_to_active_switch_by_budget(budget_list, n_iterations=10):
    """For each total budget, plot a subplot:

    - Main line: completely random sampling (10 rounds)
    - 10 branch lines: switch to active learning after round t (t=1..10)
    The x-axis ranges from 0 to n_iterations, where 0 represents the performance with only the initial 2 points.
    """

    n_cases = len(budget_list)
    n_cols = min(4, n_cases)
    n_rows = int(np.ceil(n_cases / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, budget in zip(axes, budget_list):
        # Set the annotation amount per round k based on the budget (ensure total annotation does not exceed budget)
        k_case = max(1, (budget - 2) // n_iterations)

        # Collect curves from multiple experiments
        random_results = []  # Completely random
        hybrid_results = {t: [] for t in range(1, n_iterations + 1)}  # Different switch points

        for _ in range(N_EXPERIMENTS):
            # ---------- Baseline: completely random + record state at each round ----------
            X_pool = np.random.rand(N_POOL, 2)
            y_pool = get_label(X_pool)

            # Initial seeds
            pos_idx = np.where(y_pool == 1)[0]
            neg_idx = np.where(y_pool == 0)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                idx_pos = np.random.choice(pos_idx, 1)
                idx_neg = np.random.choice(neg_idx, 1)
                idx_init = np.concatenate([idx_pos, idx_neg])
            else:
                n_init = min(2, len(y_pool))
                idx_init = np.random.choice(len(y_pool), n_init, replace=False)

            X_train = X_pool[idx_init]
            y_train = y_pool[idx_init]
            X_pool_cur = np.delete(X_pool, idx_init, axis=0)
            y_pool_cur = np.delete(y_pool, idx_init, axis=0)

            model = RectangleModel()
            model.fit(X_train, y_train)

            # IoU sequence & snapshots for the random baseline
            history_rand = [calculate_iou(model.rect, TARGET_RECT)]
            snapshots = [(X_train.copy(), y_train.copy(), X_pool_cur.copy(), y_pool_cur.copy())]

            for _t in range(n_iterations):
                k_curr = min(k_case, len(X_pool_cur))
                if k_curr == 0:
                    history_rand.append(history_rand[-1])
                    snapshots.append((X_train.copy(), y_train.copy(),
                                      X_pool_cur.copy(), y_pool_cur.copy()))
                    continue

                sel_idx = np.random.choice(len(X_pool_cur), size=k_curr, replace=False)
                X_train = np.vstack([X_train, X_pool_cur[sel_idx]])
                y_train = np.concatenate([y_train, y_pool_cur[sel_idx]])
                X_pool_cur = np.delete(X_pool_cur, sel_idx, axis=0)
                y_pool_cur = np.delete(y_pool_cur, sel_idx, axis=0)

                model.fit(X_train, y_train)
                history_rand.append(calculate_iou(model.rect, TARGET_RECT))
                snapshots.append((X_train.copy(), y_train.copy(),
                                  X_pool_cur.copy(), y_pool_cur.copy()))

            random_results.append(history_rand)

            # ---------- Different switch points: share the same random prefix ----------
            for t in range(1, n_iterations + 1):
                # Extract the state after the random phase at round t
                X_train_t, y_train_t, X_pool_t, y_pool_t = snapshots[t]

                model_h = RectangleModel()
                model_h.fit(X_train_t, y_train_t)

                # Prefix: 0..t are the same as completely random
                hist_h = history_rand[:t + 1].copy()

                # From t+1 to n_iterations, use active learning
                for _step in range(t + 1, n_iterations + 1):
                    k_curr = min(k_case, len(X_pool_t))
                    if k_curr == 0:
                        hist_h.append(hist_h[-1])
                        continue

                    if model_h.rect is None:
                        sel_idx_h = np.random.choice(len(X_pool_t), size=k_curr, replace=False)
                    else:
                        d_x1 = np.abs(X_pool_t[:, 0] - model_h.rect[0])
                        d_x2 = np.abs(X_pool_t[:, 0] - model_h.rect[1])
                        d_y1 = np.abs(X_pool_t[:, 1] - model_h.rect[2])
                        d_y2 = np.abs(X_pool_t[:, 1] - model_h.rect[3])
                        dist = np.minimum.reduce([d_x1, d_x2, d_y1, d_y2])
                        sel_idx_h = np.argsort(dist)[:k_curr]

                    X_train_t = np.vstack([X_train_t, X_pool_t[sel_idx_h]])
                    y_train_t = np.concatenate([y_train_t, y_pool_t[sel_idx_h]])
                    X_pool_t = np.delete(X_pool_t, sel_idx_h, axis=0)
                    y_pool_t = np.delete(y_pool_t, sel_idx_h, axis=0)

                    model_h.fit(X_train_t, y_train_t)
                    hist_h.append(calculate_iou(model_h.rect, TARGET_RECT))

                hybrid_results[t].append(hist_h)

        # Calculate mean and standard deviation and plot (main line + branches with shading)
        x = np.arange(n_iterations + 1)
        rand_arr = np.array(random_results)
        mean_r = rand_arr.mean(axis=0)
        std_r = rand_arr.std(axis=0)

        cmap = plt.cm.viridis
        base_color = cmap(0.98)  # Main line close to yellow

        # Completely random main line + range
        ax.plot(x, mean_r, label='Random (all)', color=base_color, lw=2, marker='o', ms=4)
        ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                color=base_color, alpha=0.1)

        # Branch lines + range for different switch points
        for t in range(1, n_iterations + 1):
            curves_t = np.array(hybrid_results[t])
            mean_h = curves_t.mean(axis=0)
            std_h = curves_t.std(axis=0)
            color = cmap(t / (n_iterations + 1))
            ax.plot(x, mean_h, color=color, lw=1.5, marker='o', ms=3, label=f'switch@{t}')
            ax.fill_between(x, mean_h - std_h, mean_h + std_h,
                            color=color, alpha=0.1)

        ax.set_title(f"budget={budget}")
        ax.set_xlabel("Iteration")
        ax.set_xticks(np.arange(n_iterations + 1))

    # Hide extra subplots
    for ax in axes[n_cases:]:
        ax.axis('off')

    axes[0].set_ylabel("IoU")
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle("Random -> Active switch under different budgets", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



