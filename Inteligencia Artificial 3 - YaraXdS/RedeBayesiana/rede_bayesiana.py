import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
np.random.seed(0)

# CPTs (variáveis binárias: 0/1)
CPT = {}
CPT['C'] = np.array([0.8, 0.2])    # P(C=0), P(C=1)
CPT['V|C'] = {0: np.array([0.9,0.1]), 1: np.array([0.4,0.6])}
CPT['S'] = np.array([0.85, 0.15])

# tabela P(A=1 | C,V,S)
A_table = {}
for C in (0,1):
    for V in (0,1):
        for S in (0,1):
            p = 0.01
            if C == 1: p += 0.08
            if V == 1: p += 0.12
            if S == 1: p += 0.15
            p = min(p, 0.95)
            A_table[(C,V,S)] = p
CPT['A|CVS'] = A_table

def all_assignments(vars_list):
    if not vars_list:
        return [{}]
    rest = all_assignments(vars_list[1:])
    res = []
    for val in (0,1):
        for r in rest:
            d = r.copy(); d[vars_list[0]] = val; res.append(d)
    return res

def joint_prob(assign):
    p = 1.0
    p *= CPT['C'][assign['C']]
    p *= CPT['V|C'][assign['C']][assign['V']]
    p *= CPT['S'][assign['S']]
    p *= CPT['A|CVS'][(assign['C'], assign['V'], assign['S'])] if assign['A'] == 1 else (1 - CPT['A|CVS'][(assign['C'], assign['V'], assign['S'])])
    return p

def query(var, evidence):
    vars_all = ['C','V','S','A']
    num = 0.0; den = 0.0
    for assign in all_assignments(vars_all):
        ok = True
        for k,v in evidence.items():
            if assign[k] != v:
                ok = False; break
        if not ok: continue
        assign_num = assign.copy(); assign_num[var] = 1
        num += joint_prob(assign_num)
        den += joint_prob(assign)
    return num / den if den > 0 else 0.0

def demo_and_save():
    p_A = query('A', {})
    p_A_given_Vbad = query('A', {'V':1})
    p_A_cv_s = query('A', {'C':1, 'S':1})

    OUT = OUT_DIR
    labels = ['P(A)', 'P(A|V=low)', 'P(A|C=bad,S=inad)']
    vals = [p_A, p_A_given_Vbad, p_A_cv_s]
    plt.figure(); plt.bar(range(len(vals)), vals); plt.xticks(range(len(vals)), labels, rotation=15)
    plt.ylabel('Probability'); plt.title('Accident probabilities under scenarios'); plt.tight_layout()
    plt.savefig(OUT / 'accident_prob_scenarios.png'); plt.close()

    # Segundo modelo: Alarm (B=Burglary, T=Tampering, A=Alarm)
    P_B = np.array([0.995, 0.005]); P_T = np.array([0.99, 0.01])
    P_A = {}
    for b in (0,1):
        for t in (0,1):
            p = 0.001 + 0.95*b + 0.6*t
            p = min(p, 0.999)
            P_A[(b,t)] = p
    total = 0.0
    for b in (0,1):
        for t in (0,1):
            total += P_B[b] * P_T[t] * P_A[(b,t)]
    plt.figure(); plt.bar([0],[total]); plt.xticks([0], ['P(Alarm)']); plt.title('Alarm marginal probability'); plt.tight_layout()
    plt.savefig(OUT / 'alarm_marginal.png'); plt.close()

    # Salva CPTs e resultados numéricos
    with open(OUT / 'cpts_and_results.txt', 'w', encoding='utf-8') as f:
        f.write('CPT C: ' + str(CPT['C']) + '\n\n')
        f.write('CPT V|C: ' + str(CPT['V|C']) + '\n\n')
        f.write('CPT S: ' + str(CPT['S']) + '\n\n')
        f.write('A table (P(A=1|C,V,S)):\n' + str(CPT['A|CVS']) + '\n\n')
        f.write('Computed probabilities:\nP(A)={:.6f}\nP(A|V=low)={:.6f}\nP(A|C=bad,S=inad)={:.6f}\nAlarm marginal={:.6f}\n'.format(p_A, p_A_given_Vbad, p_A_cv_s, total))

if __name__ == '__main__':
    demo_and_save()