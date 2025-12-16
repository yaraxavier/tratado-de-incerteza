import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent  # salva PNGs na mesma pasta
np.random.seed(0)

def normalize(a):
    s = np.sum(a)
    return a / s if s != 0 else a

def forward(obs, A, B, pi):
    """
    Algoritmo Forward com escalamento.
    obs: lista de inteiros (0..M-1)
    A: (N,N) transições
    B: (N,M) emissões
    pi: (N,) inicial
    Retorna: alpha (T,N) e log-likelihood (float)
    """
    N = A.shape[0]; T = len(obs)
    alpha = np.zeros((T, N)); c = np.zeros(T)
    alpha[0] = pi * B[:, obs[0]]
    c[0] = np.sum(alpha[0]); alpha[0] /= (c[0] + 1e-12)
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]
        c[t] = np.sum(alpha[t]); alpha[t] /= (c[t] + 1e-12)
    loglik = np.sum(np.log(c + 1e-12))
    return alpha, loglik

def backward(obs, A, B, c):
    N = A.shape[0]; T = len(obs)
    beta = np.zeros((T, N))
    beta[-1] = 1.0 / (c[-1] + 1e-12)
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, obs[t+1]] * beta[t+1, :])
        beta[t] /= (c[t] + 1e-12)
    return beta

def baum_welch(obs, N, M, max_iter=25):
    """
    Baum-Welch (EM) básico sobre uma única sequência obs (didático).
    Retorna A, B, pi e histórico de log-likelihood por iteração.
    """
    A = normalize(np.random.rand(N, N) + 0.1)
    B = normalize(np.random.rand(N, M) + 0.1)
    pi = normalize(np.random.rand(N) + 0.1)
    ll_history = []
    T = len(obs)
    for it in range(max_iter):
        # Forward com escalamento (c)
        alpha = np.zeros((T, N)); c = np.zeros(T)
        alpha[0] = pi * B[:, obs[0]]; c[0] = np.sum(alpha[0]); alpha[0] /= (c[0] + 1e-12)
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]
            c[t] = np.sum(alpha[t]); alpha[t] /= (c[t] + 1e-12)
        beta = backward(obs, A, B, c)
        loglik = np.sum(np.log(c + 1e-12)); ll_history.append(loglik)

        xi = np.zeros((T-1, N, N)); gamma = np.zeros((T, N))
        for t in range(T-1):
            denom = np.sum(alpha[t] * np.dot(A, (B[:, obs[t+1]] * beta[t+1])))
            for i in range(N):
                numer = alpha[t, i] * A[i, :] * B[:, obs[t+1]] * beta[t+1, :]
                xi[t, i, :] = numer / (denom + 1e-12)
            gamma[t, :] = np.sum(xi[t], axis=1)
        gamma[-1, :] = alpha[-1, :]

        # Reestimar
        pi = gamma[0] / (np.sum(gamma[0]) + 1e-12)
        for i in range(N):
            for j in range(N):
                A[i, j] = np.sum(xi[:, i, j]) / (np.sum(gamma[:-1, i]) + 1e-12)
        for i in range(N):
            for k in range(M):
                s = 0.0
                for t in range(T):
                    if obs[t] == k:
                        s += gamma[t, i]
                B[i, k] = s / (np.sum(gamma[:, i]) + 1e-12)
        # normalizar linhas
        for i in range(N):
            A[i] = A[i] / (np.sum(A[i]) + 1e-12)
            B[i] = B[i] / (np.sum(B[i]) + 1e-12)

    return A, B, pi, ll_history

def demo_and_save():
    # Símbolos discretos: 3 símbolos (0,1,2)
    seq_sim = [0,0,1,0,2,0,1,0]
    seq_nao = [2,1,2,2,1,2,0,2]
    N, M = 2, 3

    A_s, B_s, pi_s, ll_s = baum_welch(seq_sim, N, M, max_iter=25)
    A_n, B_n, pi_n, ll_n = baum_welch(seq_nao, N, M, max_iter=25)

    # Log-likelihood cross-eval (comparação 'Sim' x 'Não')
    _, ll_sim_under_sim = forward(seq_sim, A_s, B_s, pi_s)
    _, ll_sim_under_nao = forward(seq_sim, A_n, B_n, pi_n)
    _, ll_nao_under_sim = forward(seq_nao, A_s, B_s, pi_s)
    _, ll_nao_under_nao = forward(seq_nao, A_n, B_n, pi_n)

    OUT = OUT_DIR
    # Gráfico comparação
    labels = ['seq_sim|sim', 'seq_sim|nao', 'seq_nao|sim', 'seq_nao|nao']
    vals = [ll_sim_under_sim, ll_sim_under_nao, ll_nao_under_sim, ll_nao_under_nao]
    plt.figure(); plt.bar(range(len(vals)), vals); plt.xticks(range(len(vals)), labels, rotation=20)
    plt.title('Log-likelihood comparison (higher -> more likely)'); plt.tight_layout()
    plt.savefig(OUT / 'log_likelihood_comparison.png'); plt.close()

    # Histórico do EM
    plt.figure(); plt.plot(ll_s, marker='o'); plt.title("HMM 'Sim' - Log-likelihood por iteração"); plt.xlabel("Iter"); plt.ylabel("LL"); plt.tight_layout()
    plt.savefig(OUT / 'hmm_sim_training_ll.png'); plt.close()

    plt.figure(); plt.plot(ll_n, marker='o'); plt.title("HMM 'Nao' - Log-likelihood por iteração"); plt.xlabel("Iter"); plt.ylabel("LL"); plt.tight_layout()
    plt.savefig(OUT / 'hmm_nao_training_ll.png'); plt.close()

    # Salva parâmetros
    with open(OUT / 'trained_params.txt', 'w', encoding='utf-8') as f:
        f.write('A_sim:\n' + str(A_s) + '\nB_sim:\n' + str(B_s) + '\npi_sim:\n' + str(pi_s) + '\n\n')
        f.write('A_nao:\n' + str(A_n) + '\nB_nao:\n' + str(B_n) + '\npi_nao:\n' + str(pi_n) + '\n\n')

if __name__ == '__main__':
    demo_and_save()