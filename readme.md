# 🧠 Inteligência Artificial 3 – Tratando Incerteza

Este repositório contém a implementação prática dos principais **modelos probabilísticos para tratamento de incerteza**, desenvolvidos como parte do portfólio da disciplina **FGA0221 – Inteligência Artificial**, da **Universidade de Brasília (UnB)**.

O trabalho foca na **implementação manual dos algoritmos**, priorizando o entendimento matemático e conceitual em vez do uso de bibliotecas prontas.

---

## 👩‍🎓 Informações Acadêmicas

* **Aluno(a):** Yara Xavier de Sousa
* **Matrícula:** 211015473
* **Disciplina:** FGA0221 – Inteligência Artificial
* **Tema:** Tratamento de Incerteza
* **Professor:** Fabiano Araujo Soares

---

## 🎯 Objetivo do Projeto

Implementar e analisar algoritmos clássicos de **Inteligência Artificial Probabilística**, capazes de lidar com **incertezas, ruídos e estados ocultos**, aplicados a diferentes contextos.

O projeto demonstra domínio em:

* Probabilidade e Estatística
* Álgebra Linear
* Modelos Estocásticos
* Inferência e Otimização

---

## 📂 Estrutura do Projeto

```text
Inteligencia Artificial 3 - YaraXdS/
│── build_docs_and_zip.py
│
├── EKF/
│   └── ekf_volatility.py
│
├── HMM/
│   └── hmm_reconhecimento.py
│
├── RedeBayesiana/
│   └── rede_bayesiana.py
```

---

## 🧩 Módulos Implementados

### 🔹 1. Hidden Markov Model (HMM)

📁 `HMM/hmm_reconhecimento.py`

* Implementação manual de **Modelos Ocultos de Markov**
* Aplicação em **reconhecimento / inferência de estados ocultos**
* Uso do algoritmo **Baum-Welch** para treinamento
* Cálculo explícito de:

  * Probabilidades de transição
  * Probabilidades de emissão
  * Distribuição inicial

➡️ Destaca domínio em **inferência probabilística** e **otimização iterativa**.

---

### 🔹 2. Filtro de Kalman Estendido (EKF)

📁 `EKF/ekf_volatility.py`

* Implementação do **Extended Kalman Filter**
* Aplicação na modelagem de **volatilidade**
* Tratamento de sistemas **não lineares**
* Etapas principais:

  * Predição do estado
  * Linearização via Jacobiano
  * Atualização com medições ruidosas

➡️ Demonstra forte base em **álgebra linear**, **probabilidade** e **modelos dinâmicos**.

---

### 🔹 3. Redes Bayesianas

📁 `RedeBayesiana/rede_bayesiana.py`

* Construção manual de uma **Rede Bayesiana**
* Modelagem de dependências condicionais entre variáveis
* Inferência probabilística baseada em evidências

➡️ Evidencia compreensão sólida de **causalidade probabilística**.

---

## 🛠️ Script Auxiliar

### 🔸 build_docs_and_zip.py

* Script utilizado para **organização, documentação e empacotamento** do projeto
* Facilita a geração de arquivos finais para entrega acadêmica

---

## 🚀 Como Executar

1. Clone o repositório:

```bash
git clone <url-do-repositorio>
```

2. Acesse o diretório:

```bash
cd Inteligencia\ Artificial\ 3\ -\ YaraXdS
```

3. Execute qualquer módulo individualmente:

```bash
python HMM/hmm_reconhecimento.py
python EKF/ekf_volatility.py
python RedeBayesiana/rede_bayesiana.py
```

---

## 🏆 Competências Demonstradas

* Implementação matemática de algoritmos clássicos de IA
* Tratamento explícito de incerteza e ruído
* Raciocínio probabilístico avançado
* Capacidade para **Pesquisa e Desenvolvimento (P&D)**
* Código claro, organizado e conceitualmente fundamentado

---

## 📌 Observações Finais

Este projeto representa um **nível técnico avançado** dentro da disciplina de Inteligência Artificial, sendo adequado para uso em **portfólio acadêmico**, **apresentações técnicas** e **processos seletivos na área de IA e Ciência de Dados**.

---

📍 *Universidade de Brasília – Faculdade do Gama*
