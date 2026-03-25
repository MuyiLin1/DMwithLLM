import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Task 3: The Anisotropic LLM Wrapper
# ==========================================
class AnisotropicLLMWrapper:
    """
    Translates a black-box LLM's text/index prediction into the mathematical 
    theta and V matrix required by our geometric routing algorithm.
    """
    def __init__(self, action_features, lam=1.0, L_sq=10000.0):
        self.action_features = action_features
        self.lam = lam
        self.L_sq = L_sq # Massive confidence multiplier

    def get_prediction_and_uncertainty(self, chosen_action_idx, context_x):
        # 1. Grab the feature vector (z) of the action the LLM chose
        z = self.action_features[chosen_action_idx]
        
        # 2. Fake Theta: The LLM strongly believes the reward lies in direction z
        theta_llm = z.copy()
        
        # 3. Fake Uncertainty: x^T V^{-1} x using Sherman-Morrison formula
        # This mathematically simulates a V matrix that is incredibly confident 
        # in the LLM's chosen direction, but highly uncertain elsewhere.
        x_norm_sq = np.dot(context_x, context_x)
        z_norm_sq = np.dot(z, z)
        xz_dot = np.dot(context_x, z)
        
        term1 = (1.0 / self.lam) * x_norm_sq
        term2 = (self.L_sq / (self.lam * (self.lam + self.L_sq * z_norm_sq))) * (xz_dot ** 2)
        uncertainty_llm = term1 - term2
        
        return theta_llm, uncertainty_llm

# ==========================================
# 2. Standard LinUCB Functions
# ==========================================
def get_linucb_uncertainty(V, x):
    vx = np.linalg.solve(V, x)
    return float(x @ vx)

def linucb_proposal(V, b, X, delta=0.05, sigma=0.1):
    theta_hat = np.linalg.solve(V, b)
    bonus = sigma * np.sqrt(2.0 * np.log(1.0 / delta)) + 1.0

    vals = []
    for x in X:
        mean = float(theta_hat @ x)
        rad = np.sqrt(get_linucb_uncertainty(V, x))
        vals.append(mean + bonus * rad)
    return int(np.argmax(vals))

def ridge_update(V, b, x, r):
    return V + np.outer(x, x), b + r * x

# ==========================================
# 3. Task 1: The Deterministic Experiment
# ==========================================
def run_trial(T=100, sigma=0.1, lam=1.0, delta=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # The true hidden reward parameter
    theta_star = np.array([1.0, 0.0])

    def step_reward(x):
        return float(x @ theta_star + rng.normal(0.0, sigma))

    # Action Feature Space
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0])
    X = [x1, x2]
    
    # Initialize our Anisotropic Wrapper for the LLM
    llm_wrapper = AnisotropicLLMWrapper(action_features=X, lam=lam, L_sq=10000.0)

    # Algorithm A: Standard LinUCB Only (Baseline)
    V_A = lam * np.eye(2)
    b_A = np.zeros(2)

    # Algorithm B: Minimum Uncertainty Router (Our Algorithm)
    V_B = lam * np.eye(2)
    b_B = np.zeros(2)

    cum_reward_A = 0.0
    cum_reward_B = 0.0
    avg_reward_A = np.zeros(T)
    avg_reward_B = np.zeros(T)

    for t in range(1, T + 1):
        # We add slight noise to the context vectors to force the 
        # algorithm to evaluate geometry every round
        noise1, noise2 = rng.uniform(-0.1, 0.1, 2)
        current_context_x1 = np.array([1.0 + noise1, 0.0])
        current_context_x2 = np.array([0.0, 1.0 + noise2])
        current_X = [current_context_x1, current_context_x2]

        # --- Baseline: Standard LinUCB ---
        aA = linucb_proposal(V_A, b_A, current_X, delta=delta, sigma=sigma)
        rA = step_reward(current_X[aA])
        V_A, b_A = ridge_update(V_A, b_A, current_X[aA], rA)
        cum_reward_A += rA
        avg_reward_A[t - 1] = cum_reward_A / t

        # --- Our Algorithm: Deterministic Minimum Uncertainty Router ---
        
        # 1. Ask the Black-Box LLM what it wants to do (Assume it smartly picks Action 0)
        llm_choice_idx = 0 
        
        # 2. Use the Wrapper to translate the LLM's choice into an uncertainty score
        _, u_llm = llm_wrapper.get_prediction_and_uncertainty(llm_choice_idx, current_X[llm_choice_idx])

        # 3. Ask LinUCB what it wants to do, and get its uncertainty score
        a_lin = linucb_proposal(V_B, b_B, current_X, delta=delta, sigma=sigma)
        u_lin = get_linucb_uncertainty(V_B, current_X[a_lin])

        # 4. William's Deterministic Rule: Route to the one with the smallest uncertainty!
        if u_llm <= u_lin:
            aB = llm_choice_idx
        else:
            aB = a_lin

        # 5. Execute action and update ONLY the LinUCB background model
        rB = step_reward(current_X[aB])
        V_B, b_B = ridge_update(V_B, b_B, current_X[aB], rB)
        cum_reward_B += rB
        avg_reward_B[t - 1] = cum_reward_B / t

    return avg_reward_A, avg_reward_B

def experiment(n_trials=50, T=150, seed=42):
    rng = np.random.default_rng(seed)
    avgA = np.zeros(T)
    avgB = np.zeros(T)

    print("Running experiments...")
    for _ in range(n_trials):
        A, B = run_trial(T=T, rng=rng)
        avgA += A
        avgB += B

    avgA /= n_trials
    avgB /= n_trials

    # Plotting the results
    t = np.arange(1, T + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(t, avgA, label="Standard LinUCB", color='red', linestyle='--')
    plt.plot(t, avgB, label="Min-Uncertainty Router (Our Algorithm)", color='blue', linewidth=2)
    
    plt.xlabel("Time Step (t)", fontsize=12)
    plt.ylabel("Average Cumulative Reward", fontsize=12)
    plt.title("Reward Multiplication via Deterministic Geometric Routing", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig("reward_multiplication.png", dpi=300)
    print("Experiment complete. Saved plot to 'reward_multiplication.png'.")

if __name__ == "__main__":
    experiment()