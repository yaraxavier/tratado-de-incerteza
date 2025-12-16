"""
build_docs_and_zip.py
Gera os três DOCX (HMM, Rede Bayesiana, EKF) com conteúdo acadêmico (sem código),
coloca cada DOCX dentro da pasta correspondente e cria um ZIP final com as três pastas.
"""

from docx import Document
from pathlib import Path
import zipfile, shutil

ROOT = Path.cwd()
HMM_DIR = ROOT / "HMM"
BN_DIR = ROOT / "Rede_Bayesiana"
EKF_DIR = ROOT / "EKF"
OUT_ZIP = ROOT / "probabilistic_methods_package.zip"

def create_docx(path, title, sections):
    doc = Document()
    doc.add_heading(title, level=1)
    for heading, body in sections:
        doc.add_heading(heading, level=2)
        for p in body.split("\n\n"):
            doc.add_paragraph(p)
    doc.save(path)

# Conteúdos (acadêmicos) — sem código
hmm_sections = [
    ("Introdução e fundamentação teórica", "Este documento descreve o reconhecimento de palavras utilizando Modelos de Markov Ocultos (HMM). Apresenta a fundamentação matemática dos HMM, definindo estados ocultos, observações discretas e estrutura de parâmetros (A, B, π)."),
    ("Definição do modelo (A, B, π e observações)", "A matriz de transição A, matriz de emissões B e distribuição inicial π são definidas formalmente. Observações são símbolos discretos de um vocabulário finito."),
    ("Descrição do treinamento, inferência e algoritmo Forward", "Treinamento: Baum-Welch (EM). Inferência: algoritmo Forward com escalamento para estabilidade numérica. A decodificação pode ser feita com Viterbi (não detalhado aqui)."),
    ("Explicação dos cálculos de log-likelihood", "A log-verossimilhança é obtida pela soma dos log dos fatores de escalamento c_t, mitigando underflow, resultando em log P(O|λ) = Σ log c_t."),
    ("Tabelas, gráficos e diagramas explicativos", "Gráficos do histórico de treinamento e comparação de log-verossimilhanças foram gerados (arquivo PNG na pasta HMM)."),
    ("Comparação de probabilidades das palavras (\"Sim\" x \"Não\")", "A comparação é realizada avaliando log-likelihoods de uma mesma sequência sob modelos distintos; a maior verossimilhança indica o modelo mais provável."),
    ("Conclusão acadêmica completa", "HMMs são modelos probabilísticos robustos para sequências temporais discretas. Para aplicações reais recomenda-se validação cruzada, regularização e modelos com vocabulário e estados adequados.")
]

create_docx(HMM_DIR / "HMM_academic.docx", "Reconhecimento de Palavras com Modelos de Markov Ocultos (HMM)", "\n\n".join([f"{h}\n\n{b}" for h,b in hmm_sections]))

bn_sections = [
    ("Introdução", "Este documento apresenta Modelagem Probabilística com Redes Bayesianas aplicada a risco de acidentes e a um sistema de alarme."),
    ("Definição das variáveis C, V, S e A", "Definição: C (Condições Climáticas), V (Visibilidade), S (Sinalização), A (Acidente). Todas consideradas binárias neste modelo didático."),
    ("Grafo causal desenhado e ilustrado", "Topologia sugerida: C->V, C->A, V->A, S->A. O grafo captura que clima afeta visibilidade, e essas variáveis, junto com sinalização, influenciam o risco de acidente."),
    ("Tabelas CPT corretamente formatadas", "As CPTs são especificadas numericamente: P(C), P(V|C), P(S) e P(A|C,V,S). Em redes pequenas cada tabela pode ser inspecionada diretamente."),
    ("Explicação da inferência causal e diagnóstica", "Inferência causal e diagnóstica é feita por enumeração em redes pequenas; métodos exatos/approx existem para redes maiores (junction tree, variational, MCMC)."),
    ("Gráficos das probabilidades e cenários", "Os gráficos com P(A) sob diferentes evidências foram gerados (PNG na pasta Rede_Bayesiana) para mostrar impacto das variáveis."),
    ("Explicação da segunda Rede Bayesiana (Sistema de Alarme)", "Segundo modelo: B (Burglary), T (Tampering), A (Alarm). O documento analisa sensibilidade e taxa de falsos positivos do sistema."),
    ("Conclusão técnica e sólida", "Redes Bayesianas permitem modelagem clara de incerteza e diagnóstico causal; a qualidade das CPTs e dados influencia diretamente a utilidade do modelo.")
]

create_docx(BN_DIR / "RedeBayesiana_academic.docx", "Modelagem Probabilística com Redes Bayesianas", "\n\n".join([f"{h}\n\n{b}" for h,b in bn_sections]))

ekf_sections = [
    ("Introdução ao problema da volatilidade oculta", "Explanação do problema de volatilidade latente em séries financeiras e a necessidade de estimadores dinâmicos."),
    ("Modelo matemático completo (estado, medição, processo e Jacobiano)", "Estado: x_k = log-volatilidade. Processo linear: x_{k+1}=φ x_k + w_k. Observação: y_k ~ N(0, exp(x_k)). A medição é não-linear em x; Jacobiano H_k = d/dx exp(x) = exp(x)."),
    ("Linearização via Vega como matriz H_k", "A linearização por Taylor (jacobiano) é usada no EKF; nesta abordagem utilizamos z=y^2 cuja esperança condicional E[z|x]=exp(x) para facilitar atualização."),
    ("Gráfico da volatilidade verdadeira vs estimada", "Gráficos comparativos encontrados na pasta EKF ilustram desempenho do EKF em simulação sintética."),
    ("Gráficos adicionais: erro, convergência, estabilidade", "Diagramas de erro e de convergência do estado e covariância permitem avaliar estabilidade do filtro e ajustar parâmetros."),
    ("Conclusão detalhada e fundamentos teóricos", "EKF é eficaz em muitos cenários; contudo, linearizações introduzem erro e validação é essencial em regimes fortemente não-lineares.")
]

create_docx(EKF_DIR / "EKF_academic.docx", "Filtro de Kalman Estendido aplicado ao Rastreamento de Volatilidade", "\n\n".join([f"{h}\n\n{b}" for h,b in ekf_sections]))

# Cria ZIP contendo as três pastas (sem README)
if OUT_ZIP.exists():
    OUT_ZIP.unlink()
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['HMM', 'Rede_Bayesiana', 'EKF']:
        for root, _, files in __import__('os').walk(folder):
            for f in files:
                full = __import__('os').path.join(root, f)
                arc = __import__('os').path.relpath(full, '.')
                zf.write(full, arc)

print("DOCX criados nas pastas e ZIP gerado em:", OUT_ZIP)