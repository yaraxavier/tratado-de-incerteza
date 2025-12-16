import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
np.random.seed(0)

def generate_synthetic(T=300, seed=42):
    np.random.seed(seed)
    phi = 0.98; q = 0.1
    x = np.zeros(T); x[0] = np.log(0.04)
    for k in range(1, T):
        x[k] = phi * x[k-1] + q * np.random.randn()
    y = np.array([np.random.randn() * np.sqrt(np.exp(xk)) for xk in x])
    return x, y

def ekf(y, phi=0.99, q=0.05, r=1e-6):
    T = len(y)
    x_hat = np.zeros(T); P = np.zeros(T)
    x_hat[0] = np.log(np.var(y) + 1e-6); P[0] = 1.0
    for k in range(1, T):
        # Prediction
        x_pred = phi * x_hat[k-1]; P_pred = phi * P[k-1] * phi + q*q
        # Use z = y^2 as measurement with E[z|x]=exp(x)
        z = y[k] * y[k]
        h = lambda x: np.exp(x)
        H = np.exp(x_pred)  # Jacobiano dh/dx at x_pred
        R = 2 * (np.exp(2 * x_pred)) + r  # approx Var(y^2)
        S = H * P_pred * H + R
        K = (P_pred * H) / (S + 1e-12)
        x_hat[k] = x_pred + K * (z - h(x_pred))
        P[k] = (1 - K * H) * P_pred
    return x_hat, P

def demo_and_save():
    x_true, y = generate_synthetic(T=300, seed=42)
    x_est, P = ekf(y, phi=0.99, q=0.05, r=1e-6)
    vol_true = np.sqrt(np.exp(x_true)); vol_est = np.sqrt(np.exp(x_est))

    plt.figure(); plt.plot(vol_true); plt.plot(vol_est, linestyle='--'); plt.title('Volatility: true vs estimated (EKF)')
    plt.xlabel('Time'); plt.ylabel('Volatility (std)'); plt.legend(['true', 'estimated']); plt.tight_layout(); plt.savefig(OUT_DIR / 'volatility_true_vs_est.png'); plt.close()

    error = vol_true - vol_est
    plt.figure(); plt.plot(error); plt.title('Estimation error (true - estimated)'); plt.xlabel('Time'); plt.ylabel('Error'); plt.tight_layout(); plt.savefig(OUT_DIR / 'estimation_error.png'); plt.close()

    plt.figure(); plt.plot(x_true); plt.plot(x_est, linestyle='--'); plt.title('EKF: state (log-volatility) true vs estimated')
    plt.xlabel('Time'); plt.ylabel('Log-volatility'); plt.legend(['x_true','x_est']); plt.tight_layout(); plt.savefig(OUT_DIR / 'ekf_state_convergence.png'); plt.close()

    with open(OUT_DIR / 'ekf_results.txt', 'w', encoding='utf-8') as f:
        f.write('Final estimated state (last 10):\n' + str(x_est[-10:]) + '\nFinal cov (last 10):\n' + str(P[-10:]))

if __name__ == '__main__':
    demo_and_save()