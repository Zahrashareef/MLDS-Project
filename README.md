# Adaptive Scaffolding in Intelligent Tutoring Systems

This repository contains the code, trained models and experimental artefacts produced for my MSc Machine Learning and Data Science project on **adaptive scaffolding in Intelligent Tutoring Systems (ITSs)**.

The project investigates how student interaction histories can be modelled using **Bayesian Knowledge Tracing (BKT)** and **Long Short-Term Memory (LSTM)** networks, before using **Deep Reinforcement Learning (DRL)** to decide whether a learner should attempt a problem independently or receive instructional support.

The framework is informed by the educational concept of the **Zone of Proximal Development (ZPD)** and evaluates adaptive scaffolding strategies using a semi-synthetic student-learning environment.

---

# Repository Contents

## Main Notebook

### `MLDS_code.ipynb`

This is the main project notebook and contains the complete experimental pipeline.

It includes:

- data loading and preprocessing
- exploratory data analysis
- student-disjoint train, validation and test splitting
- Bayesian Knowledge Tracing
- LSTM learner modelling
- comparison of alternative LSTM architectures
- construction of the hybrid learner state
- leakage checks
- historical support analysis
- propensity and common-support diagnostics
- student-dynamics modelling
- construction of the semi-synthetic scaffolding environment
- Double Deep Q-Network training
- Dueling Double DQN experiments
- rule-based policy baselines
- sensitivity analysis
- held-out policy evaluation
- scaffold-fading analysis
- researcher-practitioner face-validity analysis

The notebook is intended to be read and executed sequentially from top to bottom.

---

# Saved Bayesian Knowledge Tracing Model

### `bkt_baseline_1fit.pkl`

Saved fitted **Bayesian Knowledge Tracing (BKT)** model.

BKT is used as an interpretable learner model for estimating the evolving mastery of individual mathematical skills from previous student interactions.

Only learner-state information available before the current interaction is used when constructing later scaffold-decision features.

---

# Saved LSTM Learner Models

Several LSTM architectures were developed and compared during the learner-modelling stage of the project.

### `bkt_enhanced_lstm_best.keras`

Saved BKT-enhanced LSTM model combining sequential learner behaviour with BKT-derived mastery estimates.

### `network_enhanced_lstm_best.keras`

Saved LSTM variant incorporating information derived from relationships between skills in the constructed skill network.

### `problem_ability_lstm_best.keras`

Saved LSTM variant incorporating problem-level ability or difficulty information.

### `problem_embedding_lstm_best.keras`

Saved problem-aware LSTM using learned problem embeddings to model variation between individual learning items.

### `question_content_lstm_best.keras`

Saved LSTM variant incorporating question or problem-content information.

These files preserve the best checkpoint from the corresponding learner-model experiments so that the trained models can be retained without repeating the full training process.

---

# Student Dynamics and Simulation Models

### `base_student_dynamics_model.keras`

Saved **action-agnostic student-dynamics model**.

This model predicts the learner's future same-skill independent-success trajectory without taking the current scaffold action as an input.

It provides an empirical anchor for the student simulator but is **not interpreted as a causal no-support model**.

### `action_conditioned_student_simula...`

The repository also contains saved artefacts from the action-conditioned student-simulation experiments conducted during development.

These models were used while investigating how scaffold actions could be represented within the simulated learner environment.

Because historical hint use in ASSISTments is observational rather than randomly assigned, historical supported and unsupported interactions are not treated as directly identified causal effects.

---

# Deep Reinforcement Learning Models

### `best_scaffolding_double_dqn.keras`

Saved final **Standard Double Deep Q-Network (Double DQN)** used for adaptive scaffold decision-making.

The agent observes the current learner state and chooses between:

- an independent attempt
- providing instructional support

The reward function is designed to encourage future independent success while penalising unnecessary scaffolding.

### `best_dueling_double_dqn.weights.h5`

Saved weights for the **Dueling Double DQN** architecture investigated as an alternative reinforcement-learning model.

The Standard and Dueling architectures were compared during development before the final policy was frozen for held-out evaluation.

---

# Skill Network Files

### `skill_network_nodes.csv`

Contains the mathematical skills represented as nodes in the constructed skill network.

### `skill_network_edges.csv`

Contains relationships or transitions identified between skills.

### `skill_network_nodes_connected.csv`

Contains the representation of skill nodes belonging to the connected skill network.

The skill network was explored as an additional source of information for modelling relationships between mathematical concepts and improving learner-state representations.

---

# Researcher-Practitioner Face-Validity Review

### `RESEARCHER_PRACTITIONER_RESP...`

Contains the completed researcher-practitioner scaffold decisions used in the final face-validity analysis.

### `practitioner_review_MATERIALS.zip`

Contains the anonymised practitioner-facing materials generated for the review.

The review used **24 frozen held-out learner cases**.

For each case, the previous interactions of a learner on the same mathematical skill were presented and a judgement was made about whether the learner should:

- attempt the next problem independently, or
- receive instructional support

The decisions produced by the computational policies were hidden while these judgements were made.

The completed responses were subsequently compared with the decisions generated by the computational policies using descriptive measures including:

- percentage agreement
- Cohen's kappa
- action distributions
- case-level comparisons

This component is treated as a **researcher-practitioner face-validity review** rather than an independent external validation study.

---

# Dataset

The project uses educational interaction data from the **FoundationalASSIST / ASSISTments** dataset.

The dataset contains information relating to:

- student identifiers
- problem identifiers
- mathematical skills
- first-attempt performance
- hint use
- answer viewing
- interaction timestamps
- historical learner behaviour
- problem and skill information

The main learner-modelling outcome is `discrete_score`, interpreted in this project as **independent first-try success**.

Historical hints and answer views are treated as **observational support behaviour**.

The raw ASSISTments dataset is not included in this repository.

---

# Project Pipeline

The overall experimental pipeline is:

```text
ASSISTments interaction data
            |
            v
Data cleaning and preprocessing
            |
            v
Student-disjoint train / validation / test split
            |
            +---------------------------+
            |                           |
            v                           v
Bayesian Knowledge              LSTM learner models
Tracing (BKT)                           |
            |                           |
            +-------------+-------------+
                          |
                          v
                 Hybrid learner state
                          |
                          v
             Historical support analysis
                          |
                          v
         Propensity and overlap diagnostics
                          |
                          v
             Student-dynamics model
                          |
                          v
        ZPD-inspired semi-synthetic
             learning environment
                          |
                          v
               Double DQN policy
                          |
                +---------+---------+
                |                   |
                v                   v
          DRL policies       Rule-based baselines
                |                   |
                +---------+---------+
                          |
                          v
             Held-out policy evaluation
                          |
                          v
              Scaffold-fading analysis
                          |
                          v
       Researcher-practitioner comparison
```

---

# Methodology Overview

## Bayesian Knowledge Tracing

Bayesian Knowledge Tracing provides an interpretable estimate of a learner's evolving mastery of individual skills.

The model updates the learner's estimated knowledge state sequentially using their previous interactions.

---

## Problem-Aware LSTM

A Long Short-Term Memory network is used to capture longer-term temporal patterns in learner behaviour.

The final learner modelling framework incorporates information including:

- previous learner performance
- BKT mastery
- recent interaction history
- skill information
- problem information
- temporal features
- problem-specific embeddings

The model predicts the probability of **independent first-try success**.

---

## Hybrid Learner State

The BKT and LSTM models are combined with historical learner information to construct a state representation for scaffold decisions.

The state is designed to be **leakage-safe**.

Information that would only become available after the current decision is deliberately excluded, including current:

- `discrete_score`
- hint use
- answer viewing
- post-interaction BKT mastery

This ensures that the reinforcement-learning policy only receives information that could realistically be available when a scaffold decision is made.

---

# Historical Support and Propensity Analysis

Historical hint use cannot automatically be interpreted as evidence that hints caused later student performance.

Students experiencing difficulty may naturally be more likely to request support, creating **confounding by indication**.

The project therefore investigates historical support behaviour using propensity and common-support diagnostics.

These analyses examine:

- the probability of historical support given the learner state
- overlap between supported and unsupported observations
- inverse-probability weighting diagnostics
- measured covariate balance

The analysis is used to understand the observational support process rather than claim a directly identified causal effect of historical hints.

---

# ZPD-Inspired Scaffolding Framework

The project is informed by the educational concept of the **Zone of Proximal Development (ZPD)**.

It does **not** claim to directly measure a student's psychological ZPD.

Instead, predicted independent success is used to construct a computational ZPD-inspired proxy.

The scaffold-intensity function used within the semi-synthetic environment is:

\[
Z(q)=16q^2(1-q)^2
\]

where \(q\) represents predicted independent success.

The function is:

- low when success is very unlikely
- highest in the intermediate region
- low when independent success is already very likely

This represents the intuition that support may be most useful for learners who are between clear inability and clear independent mastery.

---

# Semi-Synthetic Student Environment

The observational dataset does not provide randomly assigned scaffold interventions.

It is therefore not possible to directly observe both:

> what would happen if support was provided

and

> what would happen if support was withheld

for the same learner in the same state.

The project therefore constructs an explicitly **semi-synthetic sequential environment**.

Student progression is anchored using empirical learner-dynamics predictions while the effect of instructional support is introduced through explicit assumptions.

These scaffold-response assumptions are treated as sensitivity parameters rather than estimated causal treatment effects.

---

# Deep Reinforcement Learning

A **Double Deep Q-Network (Double DQN)** is trained to make sequential scaffold decisions.

At each decision point, the agent observes the learner state and selects an action.

The final binary action space consists of:

```text
0 = Independent attempt
1 = Provide support
```

The policy aims to balance:

- future independent learner success
- appropriate scaffold provision
- avoidance of unnecessary support

The reinforcement-learning model is compared against simpler and more interpretable policies.

---

# Baseline Policies

The DRL policy is evaluated alongside baseline strategies including approaches such as:

- always independent
- always scaffold
- random action selection
- uncertainty-based scaffolding
- a simple ZPD-inspired heuristic policy

These baselines make it possible to determine whether the additional complexity of reinforcement learning produces a meaningful advantage over simpler decision rules.

---

# Sensitivity Analysis

The semi-synthetic environment contains assumptions about both:

- the magnitude of the scaffold effect
- the cost of providing unnecessary support

Policies are therefore evaluated under multiple combinations of scaffold-effect and scaffold-cost assumptions.

The effect scenarios include:

```text
Weak
Moderate
Strong
```

The policies remain frozen during sensitivity testing.

This allows the robustness of the conclusions to be examined without repeatedly tuning the DRL policy to different evaluation environments.

---

# Held-Out Evaluation

The final models and policies are evaluated on **previously unseen students**.

Students are separated between training, validation and test sets by `user_id`, preventing interactions from the same learner from appearing across experimental partitions.

The held-out test cohort is not used for model or policy selection.

Policy evaluation considers measures including:

- mean reward per decision
- scaffold rate
- simulated independent-success rate
- final simulated learner independence
- scaffold behaviour across learner-independence levels

---

# Scaffold Fading

Adaptive scaffolding should not simply provide support whenever possible.

An important objective is to **fade support as learners become increasingly capable of working independently**.

The final analysis therefore examines scaffold rates across simulated learner-independence bands.

This is used to investigate whether policies reduce unnecessary assistance as evidence of independent capability increases.

---

# Reproducibility

The repository contains the principal trained models and experimental artefacts produced during the project.

Steps taken to improve reproducibility include:

- fixed random seeds where appropriate
- student-disjoint dataset partitions
- separate training, validation and held-out test cohorts
- leakage-safe learner-state construction
- validation-based model selection
- frozen policies during final testing
- explicit documentation of semi-synthetic assumptions
- sensitivity analysis across alternative simulator assumptions
- preservation of trained model checkpoints

Some deep-learning operations may still produce small numerical differences depending on the software and hardware environment.

---

# File Format Guide

| Format | Purpose |
|---|---|
| `.ipynb` | Main Jupyter Notebook containing the experimental pipeline |
| `.keras` | Saved TensorFlow/Keras models |
| `.h5` | Saved neural-network weights |
| `.pkl` | Serialised Python/BKT model |
| `.csv` | Skill-network data and experimental outputs |
| `.zip` | Packaged researcher-practitioner review materials |
| `.md` | Repository documentation |

---

# Important Methodological Notes

## Historical hints are observational

Historical support use is not randomly assigned.

Raw differences between students who did and did not use hints are therefore not interpreted as causal scaffold effects.

## The ZPD measure is a proxy

The project does not claim to directly measure the psychological Zone of Proximal Development.

The ZPD-inspired formulation is a computational operationalisation used to support adaptive decision-making.

## Policy evaluation is semi-synthetic

The simulated environment combines empirical learner trajectories with explicit assumptions about scaffold effects.

Results should therefore be interpreted as policy comparisons under the specified simulation assumptions rather than direct evidence of real-world causal learning gains.

## Held-out evaluation concerns unseen learners

The final experimental split evaluates generalisation to previously unseen students within the represented educational environment.

It does not establish automatic generalisation to completely different curricula, subjects or learner populations.

---

# Limitations

The main limitations of the project include:

- observational historical support behaviour
- inability to directly identify counterfactual scaffold effects
- reliance on explicit assumptions within the semi-synthetic environment
- evaluation using one educational dataset
- limited representation of affective factors such as motivation or frustration
- a small researcher-practitioner face-validity review
- no prospective classroom deployment of the final DRL policy

---

# Future Work

Potential extensions of the work include:

- prospective evaluation with real learners
- randomised scaffold interventions
- larger practitioner evaluations
- additional scaffold types beyond binary support
- personalised scaffold selection
- incorporation of motivation and affective learner states
- evaluation across additional subjects and datasets
- offline reinforcement-learning approaches designed for observational educational data
- interpretable or constrained reinforcement-learning policies
- real-time deployment within an Intelligent Tutoring System

---

# Project Aim

The overall aim of this project is to investigate whether machine learning and sequential decision-making can help an Intelligent Tutoring System answer two important questions:

> **When should a learner receive instructional support?**

and

> **When should that support be faded so that the learner can work independently?**

Rather than optimising immediate correctness alone, the framework focuses on helping learners progress towards **successful independent problem solving**.

---

# Acknowledgements

This project uses educational interaction data from the ASSISTments ecosystem and builds on research in Intelligent Tutoring Systems, Bayesian Knowledge Tracing, student modelling, adaptive scaffolding, the Zone of Proximal Development and reinforcement learning for education.
