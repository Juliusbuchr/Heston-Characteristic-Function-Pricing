import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from scipy.integrate import quad




# Parameters for Heston model (assumed for pricing without calibration errors)
S0 = 126.45  # Initial stock price
K = 130      # Strike price
T = 3 / 12   # Time to maturity (3 months)
dt = 1 / 252 # Daily time step
n_steps = int(T / dt)
n_paths = 100000

# Heston parameters
kappa = 3
theta = 0.10
rho = 0.20
sigma = 0.50
v0 = 0.10
r = 0.0515  # Risk-free rate

# Seed for reproducibility
np.random.seed(42)

# Simulate Heston model paths
S = np.full((n_steps + 1, n_paths), S0)
v = np.full((n_steps + 1, n_paths), v0)

for t in range(1, n_steps + 1):
    z1 = np.random.normal(size=n_paths)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=n_paths)
    
    v_t = v[t - 1] + kappa * (theta - np.maximum(v[t - 1], 0)) * dt + sigma * np.sqrt(np.maximum(v[t - 1], 0)) * np.sqrt(dt) * z2
    v[t] = np.maximum(v_t, 0)
    
    S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt + np.sqrt(v[t]) * np.sqrt(dt) * z1)

log_ST = np.log(S[-1])
vT = v[-1]

log_ST_mean, log_ST_median, log_ST_skew, log_ST_kurt = np.mean(log_ST), np.median(log_ST), skew(log_ST), kurtosis(log_ST)
vT_mean, vT_median, vT_skew, vT_kurt = np.mean(vT), np.median(vT), skew(vT), kurtosis(vT)

print(f"log(ST): Mean = {log_ST_mean}, Median = {log_ST_median}, Skewness = {log_ST_skew}, Kurtosis = {log_ST_kurt}")
print(f"vT: Mean = {vT_mean}, Median = {vT_median}, Skewness = {vT_skew}, Kurtosis = {vT_kurt}")

# Plot log(ST) with normal distribution overlay
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(log_ST, bins=50, density=True, alpha=0.6, color='blue', label='Simulated log(ST)')
x = np.linspace(log_ST.mean() - 3 * log_ST.std(), log_ST.mean() + 3 * log_ST.std(), 100)
plt.plot(x, norm.pdf(x, log_ST_mean, np.std(log_ST)), 'r--', label='Normal dist')
plt.title("Distribution of log(ST)")
plt.legend()

# Plot vT
plt.subplot(1, 2, 2)
plt.hist(vT, bins=50, density=True, alpha=0.6, color='purple', label='Simulated vT')
plt.title("Distribution of vT")
plt.legend()
plt.show()

# Longstaff-Schwartz Monte Carlo with antithetic variates and multiple basis sets
n_paths_half = n_paths // 2
discount = np.exp(-r * dt)

# Function to generate basis functions
def generate_basis(S, v, K, basis_set):
    Sk = S / K
    if basis_set == 1:
        return np.vstack([np.ones_like(Sk), Sk, Sk**2, v, v**2]).T
    elif basis_set == 2:
        return np.vstack([np.ones_like(Sk), Sk, Sk**2, Sk**3, v, v**2, v**3]).T
    elif basis_set == 3:
        return np.vstack([np.ones_like(Sk), Sk, Sk**2, Sk**3, v, v**2, v**3, v * Sk]).T
    elif basis_set == 4:
        return np.vstack([np.ones_like(Sk), Sk, Sk**2, Sk**3, v, v**2, v**3, v * Sk, v**2 * Sk, v * Sk**2]).T
    elif basis_set == 5:
        return np.vstack([np.ones_like(Sk), Sk, Sk**2, Sk**3, Sk**4, v, v**2, v**3, v**4, v * Sk, v**2 * Sk, v * Sk**2]).T
    else:
        raise ValueError("Invalid basis set number")

# Function to calculate option price and confidence interval
def price_option_lsm(S_paths, v_paths, K, T, r, dt, n_paths_half, basis_set):
    n_steps = S_paths.shape[0]
    discount = np.exp(-r * dt)

    payoff = np.maximum(K - S_paths[-1, :n_paths_half], 0)
    payoff_antithetic = np.maximum(K - S_paths[-1, n_paths_half:], 0)

    cash_flow = np.copy(payoff)
    cash_flow_antithetic = np.copy(payoff_antithetic)

    for t in range(n_steps - 2, 0, -1):
        itm = payoff > 0
        itm_antithetic = payoff_antithetic > 0

        basis = generate_basis(S_paths[t, :n_paths_half][itm], v_paths[t, :n_paths_half][itm], K, basis_set)
        basis_antithetic = generate_basis(S_paths[t, n_paths_half:][itm_antithetic], v_paths[t, n_paths_half:][itm_antithetic], K, basis_set)

        Y = cash_flow[itm] * discount
        coef = np.linalg.lstsq(basis, Y, rcond=None)[0]
        continuation_value = basis @ coef

        Y_antithetic = cash_flow_antithetic[itm_antithetic] * discount
        coef_antithetic = np.linalg.lstsq(basis_antithetic, Y_antithetic, rcond=None)[0]
        continuation_value_antithetic = basis_antithetic @ coef_antithetic

        exercise = payoff[itm] > continuation_value
        exercise_antithetic = payoff_antithetic[itm_antithetic] > continuation_value_antithetic

        cash_flow[itm] = np.where(exercise, payoff[itm], cash_flow[itm] * discount)
        cash_flow_antithetic[itm_antithetic] = np.where(exercise_antithetic, payoff_antithetic[itm_antithetic], cash_flow_antithetic[itm_antithetic] * discount)

    option_price = np.mean(cash_flow) * np.exp(-r * T)
    option_price_antithetic = np.mean(cash_flow_antithetic) * np.exp(-r * T)
    american_option_price = 0.5 * (option_price + option_price_antithetic)

    all_cash_flows = np.concatenate([cash_flow, cash_flow_antithetic])
    standard_error = np.std(all_cash_flows * np.exp(-r * T)) / np.sqrt(len(all_cash_flows))
    confidence_interval = (american_option_price - 1.96 * standard_error, american_option_price + 1.96 * standard_error)
    
    return american_option_price, confidence_interval

# Price option using each basis set
basis_sets = [1, 2, 3, 4, 5]
for basis in basis_sets:
    price, interval = price_option_lsm(S, v, K, T, r, dt, n_paths_half, basis)
    print(f"Basis set {basis}: Price = {price}, 95% Confidence Interval = {interval}")


# European put comparison 
    
# Helper function for characteristic functions
def characteristic_function(phi, S0, K, T, r, kappa, theta, rho, sigma, v0, j):
    xi = np.sqrt((kappa - (rho * sigma * 1j * phi))**2 + (sigma**2) * (1j * phi + phi**2))
    if j == 1:
        num = kappa - rho * sigma * 1j * phi + xi
    else:
        num = kappa - rho * sigma * 1j * phi - xi
    denom = sigma**2 * (1j * phi + phi**2)
    term1 = np.exp(1j * phi * (np.log(S0 / K) + (r - 0.5 * v0) * T))
    term2 = np.exp((v0 * (1 - np.exp(-xi * T))) / (2 * xi))
    term3 = np.exp(kappa * theta * T * ((kappa - rho * sigma * 1j * phi) / (sigma**2) - xi / sigma**2))
    return (term1 * term2 * term3).real

# Integrand for the European put option price
def integrand(phi, S0, K, T, r, kappa, theta, rho, sigma, v0, j):
    if j == 1:
        return (np.exp(-1j * phi * np.log(K)) * characteristic_function(phi - 1j, S0, K, T, r, kappa, theta, rho, sigma, v0, 1) / (1j * phi)).real
    else:
        return (np.exp(-1j * phi * np.log(K)) * characteristic_function(phi - 1j, S0, K, T, r, kappa, theta, rho, sigma, v0, 2) / (1j * phi)).real

# European put option price in Heston model
def heston_european_put_price(S0, K, T, r, kappa, theta, rho, sigma, v0):
    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, S0, K, T, r, kappa, theta, rho, sigma, v0, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, S0, K, T, r, kappa, theta, rho, sigma, v0, 2), 0, 100)[0]
    european_call_price = S0 * np.exp(-r * T) * P1 - K * np.exp(-r * T) * P2
    european_put_price = european_call_price - S0 + K * np.exp(-r * T)
    return european_put_price

# Calculate the European put price using the Heston closed-form formula
european_put_price_heston = heston_european_put_price(S0, K, T, r, kappa, theta, rho, sigma, v0)
print(f"European Put Option Price (Heston Closed-Form): {european_put_price_heston}")