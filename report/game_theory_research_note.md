# Game theory for modeling poisoning attacks in federated learning

## Purpose and scope

This note reviews game-theory concepts and their use in research on poisoning attacks in federated learning.

> **Modeling Poisoning Attacks in Federated Learning as a Strategic Game**

It introduces the relevant concepts, modeling vocabulary, game families, and published approaches.

## Game theory: the basic idea

Game theory studies situations in which the outcome for one decision-maker depends on choices made by other decision-makers. A player is **rational** in the modeling sense when it chooses actions to optimize its stated objective (utility), given its information and expectations about other players. Rationality is an assumption for analysis; it does not mean that real attackers always calculate a mathematically perfect strategy.

For poisoning attacks in FL, the intuitive strategic loop is:

1. An attacker chooses whether, when, and how strongly to poison.
2. A defender chooses a response to received updates.
3. Each side anticipates that the other side may change its behaviour.
4. Their choices jointly determine model utility, attack success, cost, and future trust.

This is different from a non-strategic defense rule. A static rule says, for example, “reject every update whose norm exceeds a threshold.” A strategic model also asks how a defender should act when rejecting a benign but unusual update is costly, and when an attacker can lower its attack intensity to evade that threshold.

## Modeling vocabulary

### Player

A **player** is an entity that selects actions and has an objective. In poisoning literature, common players are an attacker, a defender/server, a benign client, a malicious client, or a receiving node in DFL. The player should be an actual decision-maker. A model, an aggregate, or a score can be an object in a game, but should not be called a player unless it independently chooses actions and has a utility.

### Action and strategy

An **action** is one feasible choice at a particular decision point. For example, an attacker may choose attack/no attack, attack intensity, target label, or poisoning timing; a defender may choose accept, reject, reweight, inspect, or isolate.

A **strategy** specifies what action a player will take in every situation it may encounter. It can be:

- **pure**: choose one action deterministically;
- **mixed**: choose actions according to probabilities;
- **policy**: map the current observation/state to an action, often used in repeated or Markov games.

### Utility (payoff)

A **utility** or **payoff** represents what each player tries to maximize. It converts outcomes into a common numerical preference scale. In security applications, it is usually multi-objective:

- defender utility may include clean-model performance, poisoning damage avoided, false-positive cost, communication cost, and computation cost;
- attacker utility may include attack success or degradation achieved, attack effort, and risk of being detected or isolated.

The utility is not discovered automatically from the system. It is a modeling assumption and must be made explicit when reading or comparing papers.

### Information

A game has **complete information** when players know the game structure and relevant payoffs/types. It has **incomplete or asymmetric information** when one player lacks information held by another. A typical example is that a defender cannot observe whether a strange update is malicious or caused by benign non-IID data. A hidden characteristic such as benign/malicious identity or attack family is a **type**; beliefs about types motivate Bayesian games.

### State and transition

A **state** is a compact description of information that affects future decisions. In iterative learning it may include trust, recent anomaly scores, model quality, topology, or remaining attacker budget. A **transition** describes how the state changes after players act. These ideas are needed when a game spans many rounds rather than one isolated decision.

## A game model: the minimum specification

A credible game model should explicitly define the following elements before discussing an algorithm or equilibrium.

| Element | In the thesis context | Example choices |
|---|---|---|
| Players | Actual decision-makers, not models | benign receiving node(s); malicious sending node(s) |
| State | Information that carries across rounds | local model quality, trust history, observed update statistics, topology/neighborhood, previous defense action |
| Private information / type | What one player knows but the other does not | attacker is benign/malicious; attack family; attacker budget/knowledge |
| Actions | Decisions each player can actually make | attacker: attack/no attack, intensity, target, update scaling, timing; defender: accept, reweight, reject, quarantine/isolate, request more evidence |
| Observation | What a defender can see locally | cosine distance, norm, loss change on a local validation set, trust history, neighbor degree |
| Transition | How the next round follows from current decisions | model/trust update, connectivity change, posterior belief update |
| Utility / payoff | Why each player prefers one action | defender: model utility minus poisoning damage, false-positive, communication and computation costs; attacker: attack success/damage minus effort and isolation risk |
| Solution concept | Meaning of “strategically stable” behavior | Nash equilibrium, Bayesian Nash equilibrium, Stackelberg equilibrium, or an approximate policy equilibrium |

Without these definitions, a score, threshold, or weighted-average equation is a defense rule rather than a strategic game model.

## Equilibrium and solution concepts

An equilibrium is not a universal guarantee that a system is safe. It is a formal prediction of stable behavior under a particular game’s players, utilities, information, action sets, and timing.

### Nash equilibrium

In a **Nash equilibrium**, each player's strategy is a best response to the strategies of the other players. No player can improve its own utility by changing strategy alone while the others remain fixed. It is most natural for simultaneous-move games, such as an abstracted one-round attack-defense choice.

### Bayesian Nash equilibrium

A **Bayesian Nash equilibrium** extends Nash equilibrium to hidden player types. Each player selects a strategy conditional on its own type and beliefs about other types. For poisoning, this can represent a defender that does not know whether a client is benign or malicious.

### Stackelberg equilibrium

A **Stackelberg equilibrium** represents sequential commitment. A leader commits to a strategy first; a follower observes or anticipates it and best-responds. Its use depends on a defensible timeline. It should not be applied merely because one system event happens after another: the paper must explain who commits to what and what the follower knows.

### Minimax

In a zero-sum or strictly competitive game, one player's gain equals the other's loss. A **minimax** formulation selects a defense that maximizes its outcome under the attacker’s worst plausible response. It is a useful approximation, but clean-model performance, false positives, and attack effort often mean a real poisoning game is not exactly zero-sum.

### Markov-game equilibrium

In a Markov/stochastic game, strategies are policies over state and actions generate later states. The relevant solution is an equilibrium policy profile, often approximated through dynamic programming or reinforcement learning. This makes it possible to discuss adaptation across learning rounds, but also adds computational and modeling complexity.

## Which game family fits poisoning?

### 1. Simultaneous normal-form / Nash game

Use this when attacker and defender select one action without observing the other action in that round. A simple two-player game can use attack intensity as the attacker action and accept/reject/reweight as the defender action. A Nash equilibrium is a pair of actions where neither side benefits by changing alone.

**Best use:** a small, explainable baseline model or one-round analysis.

**Limitation:** it cannot naturally represent trust accumulation, delayed attacks, or adaptation over many FL rounds.

### 2. Repeated game

Repeat the stage game for training rounds `t = 1, ..., T`. The history can change trust, future availability of a neighbor, and expected utility. This matches on-off poisoning: an attacker may behave benignly early to build trust and attack later.

**Best use:** showing why a one-shot model is insufficient.

**Limitation:** the state space grows unless the history is summarized by a compact state, such as trust and recent anomaly scores.

### 3. Bayesian game

Use this when the defender cannot observe the attacker type. Let the hidden type be, for example,

`theta_j in {benign, untargeted poisoner, backdoor poisoner, adaptive poisoner}`.

The defender begins with a prior belief and updates it using local observations. This is the right language for incomplete information: a low-quality update can arise from malicious behavior or benign non-IID data, so “suspicious” is not the same as “malicious.”

**Best use:** modeling the false-positive versus missed-attack trade-off honestly.

### 4. Stochastic / Markov game

Make the current state sufficient for future decisions:

`s_t = (trust_t, anomaly_t, local_validation_t, neighborhood_t, model_state_t)`.

Actions produce a next state through a transition rule. A Markov game is appropriate when both parties adapt repeatedly and rewards are accumulated over time.

**Best use:** a principled bridge from repeated strategic interaction to reinforcement learning.

**Limitation:** a full multi-agent Markov game can become too large for a bachelor thesis.

### 5. Stackelberg game

Stackelberg means one side commits to a policy first and the other best-responds. It is appropriate only if the timing is defensible. For example, before training the defender publishes/commits to a defense policy; the attacker observes or infers that policy and chooses an attack. The defender is then the leader and the attacker is the follower.

This is **not automatically true** merely because a defender acts after receiving an update. If the attacker sends an update first and the defender reacts only afterward, the within-round action order is attacker then defender. The thesis must state whether the strategic commitment happens before the round or whether it is modeling sequential observation instead.

**Best use:** adaptive attackers that deliberately respond to a known defense policy.

### 6. Coalitional game / Shapley value

This family assigns credit to contributors or coalitions. It is useful for contribution-aware aggregation, incentives, or detecting low-contribution participants. It is generally **not** an attacker-defender strategic game by itself because it does not define adversarial best responses.

**Best use:** a related but distinct game-theory application in DFL.

## What the existing papers contribute

### Direct strategic attack-defense models

| Paper | Game model | What it contributes | Boundary relative to this thesis |
|---|---|---|---|
| Sagduyu, Erpek, and Shi (2023), *Securing NextG Systems against Poisoning Attacks on Federated Learning* | Nash game | Explicit attack-versus-defense utilities for poisoning and client admission/rejection; analyzes equilibrium and costs | Centralized FL setting and a compact two-client interaction; useful as the clearest basic template for players/actions/payoffs |
| Jia et al. (NeurIPS 2023), *FedGame* | Minimax game | Models defender versus dynamic backdoor attacker and designs an interactive defense | Centralized FL and backdoor-specific; does not solve the local, topology-dependent DFL decision problem |
| Li et al. (2024), *Meta Stackelberg Game* | Bayesian Stackelberg Markov game | Introduces hidden/uncertain attack types, adaptive attacks, pre-training plus online adaptation, and a meta-equilibrium policy | Central server/defender and high methodological complexity; strongest conceptual source for Bayesian/Markov/Stackelberg choices, not a drop-in DFL algorithm |

### Game-theory work in DFL, but not direct adversarial games

| Paper | Game-theory mechanism | Why it matters | Why it is not the proposed strategic model |
|---|---|---|---|
| Behera and Chakraborty (2024), *pFedGame* | Two-player constant-sum cooperative weighting game between a node's own model and an aggregate peer model | Demonstrates local game-inspired aggregation in dynamic DFL | No malicious player, poisoning action, defender action, hidden type, or adversarial utility; the “players” are model alternatives |
| Dong et al. (2026), *FORCE* | Shapley-value-inspired contribution aggregation | A current DFL poisoning-defense paper using a game-theoretic contribution concept | Uses contribution valuation to identify malicious clients, not strategic attacker–defender best responses or equilibrium |

## How to read and compare game-theory papers

When reviewing a paper, record these questions instead of only recording its reported accuracy:

1. What real entities are the players?
2. Are actions simultaneous, sequential, or repeated?
3. What does each player observe, and which types/information are hidden?
4. Is the game zero-sum, general-sum, cooperative, or coalitional?
5. What is included in each utility, and what costs are ignored?
6. Which equilibrium or solver is used, and under what assumptions?
7. Is the game used for adversarial decision-making, contribution valuation, incentive allocation, or only aggregation weighting?
8. Does the evaluation actually test adaptive poisoning, or only static attacks / clean-data convergence?

### Terminology cautions

- “This is the first paper to use game theory in DFL.” pFedGame and FORCE contradict it.
- “pFedGame is a poisoning-defense game.” Its evaluated objective is aggregation/convergence, not an attacker-defender interaction.
- “Stackelberg is automatically appropriate because the defender reacts later.” The timing of policy commitment and observation must be stated.
- “Label flipping is model poisoning.” Label flipping is data poisoning; use the broader phrase **poisoning attacks** unless the scope intentionally excludes it.
- “The equilibrium is optimal in reality.” It is optimal only relative to the stated utilities, information, action sets, and behavioral assumptions.

## Literature-review presentation storyline

The game-theory presentation can be built around these questions rather than re-explaining FL/DFL:

1. Why is poisoning strategic rather than only anomalous?
2. What are the players, actions, information asymmetries, and payoffs?
3. Why does a one-shot Nash game miss multi-round trust and adaptation?
4. Why do Bayesian and Markov components match hidden attacker type and repeated training?
5. When is Stackelberg justified, and what does “leader” mean here?
6. How do FedGame, the NextG Nash model, Meta Stackelberg, pFedGame, and FORCE differ?
7. What questions remain open after comparing these approaches?

## Reading priority

1. **FedGame**: best direct, peer-reviewed reference for an adversarial strategic-defense story and a minimax formulation.
2. **Meta Stackelberg Game**: best conceptual reference for Bayesian type uncertainty, dynamic state, and adaptation; use selectively because its meta-RL implementation is complex.
3. **Sagduyu et al.**: best compact reference for explaining a basic poisoning-admission game and Nash equilibrium to an undergraduate audience.
4. **pFedGame**: use as the supplied DFL-related paper, explicitly as aggregation-focused related work.
5. **FORCE**: use as current DFL evidence that game-theoretic contribution valuation has reached Byzantine-resilient decentralized aggregation, but keep it separate from strategic-game modeling.

## Sources

1. Y. E. Sagduyu, T. Erpek, and Y. Shi, “Securing NextG Systems against Poisoning Attacks on Federated Learning: A Game-Theoretic Solution,” 2023. arXiv:2312.17164. https://arxiv.org/abs/2312.17164
2. J. Jia et al., “FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning,” *NeurIPS 2023*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a6678e2be4ce7aef9d2192e03cd586b7-Abstract.html
3. T. Li, H. Li, Y. Pan, T. Xu, Z. Zheng, and Q. Zhu, “Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks,” 2024. arXiv:2410.17431. https://arxiv.org/abs/2410.17431
4. M. R. Behera and S. Chakraborty, “pFedGame - Decentralized Federated Learning using Game Theory in Dynamic Topology,” 2024. arXiv:2410.04058. https://arxiv.org/abs/2410.04058
5. Q. Dong, Z. Dai, Y. Gao, Y. Zheng, A. Fu, and W. Susilo, “FORCE: Byzantine-Resilient Decentralized Federated Learning via Game-Theoretic Contribution Aggregation,” *IEEE Transactions on Information Forensics and Security*, 2026, 21:3182-3196. DOI: 10.1109/TIFS.2026.3674690. https://doi.org/10.1109/TIFS.2026.3674690

## Verification note

This is a focused research synthesis, not an exhaustive systematic literature review. The 2024 Meta Stackelberg paper is cited as an arXiv preprint; check its final publication status before it is used as a formal thesis bibliography entry. Read the full methods sections of any paper selected for the final thesis, especially before reproducing equations, theorem claims, or experimental settings.
