import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/home/milo/workspace/project/dacn";
const SKILL_DIR = "/home/milo/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".pptx-build");
const RUNTIME_PYTHON = "/home/milo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const FINAL_PPTX = path.join(workspaceDir, "output/pptx/game-theory-poisoning-fl-literature-review-expanded-v2.pptx");
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const font = "Fira Sans";
const P = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  navy: "#0D1B2A", ink: "#122033", slate: "#506174", teal: "#007C83", aqua: "#5CC8C8",
  gold: "#D6A84B", mist: "#F2F6F7", white: "#FFFFFF", line: "#C7D4D8",
  paleTeal: "#E8F5F5", paleGold: "#FBF4E5", paleRed: "#F8ECEC", red: "#A64646",
};
let no = 0;

function box(slide, text, left, top, width, height, style = {}) {
  const shape = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { fill: "none", width: 0 } });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: 20, color: C.ink, autoFit: "shrinkText", marginLeft: 0, marginRight: 0, marginTop: 0, marginBottom: 0, ...style };
  return shape;
}
function rect(slide, left, top, width, height, fill) {
  return slide.shapes.add({ geometry: "rect", position: { left, top, width, height }, fill, line: { fill, width: 0 } });
}
function base(title) {
  no += 1;
  const s = P.slides.add();
  s.background.fill = C.white;
  box(s, title, 74, 46, 1050, 52, { fontSize: 33, bold: true, color: C.navy });
  rect(s, 74, 112, 78, 6, C.teal);
  box(s, String(no).padStart(2, "0"), 1140, 54, 64, 30, { fontSize: 14, color: C.teal, alignment: "right", bold: true });
  return s;
}
function notes(slide, text) { slide.speakerNotes.textFrame.setText(text); }
function item(slide, title, body, left, top, width, accent = C.teal, bodyH = 58) {
  rect(slide, left, top + 8, 10, 10, accent);
  box(slide, title, left + 28, top, width - 28, 28, { fontSize: 21, bold: true, color: C.navy });
  box(slide, body, left + 28, top + 34, width - 28, bodyH, { fontSize: 17, color: C.slate });
}
function keyValue(slide, label, value, left, top, width, accent = C.teal) {
  box(slide, label.toUpperCase(), left, top, width, 20, { fontSize: 13, bold: true, color: accent });
  box(slide, value, left, top + 27, width, 58, { fontSize: 19, color: C.ink, bold: false });
}
function plain(slide, text) {
  rect(slide, 96, 618, 8, 42, C.aqua);
  box(slide, "IN PLAIN LANGUAGE", 122, 617, 190, 18, { fontSize: 13, bold: true, color: C.teal });
  box(slide, text, 122, 638, 1035, 28, { fontSize: 16, color: C.ink, italic: true });
}
function tableSlide(title, values, columns, note, firstColFill = C.paleTeal) {
  const s = base(title);
  const t = s.tables.add({ rows: values.length, columns: values[0].length, left: 64, top: 144, width: 1152, height: 478, values });
  t.styleOptions = { headerRow: true, bandedRows: true };
  t.borders.assign({ style: "solid", fill: C.line, width: 1 });
  t.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length }).assign({ fill: C.navy, textStyle: { typeface: font, fontSize: 16, color: C.white, bold: true } });
  t.cells.block({ row: 1, column: 0, rowCount: values.length - 1, columnCount: 1 }).assign({ fill: firstColFill, textStyle: { typeface: font, fontSize: 15, color: C.navy, bold: true } });
  t.cells.block({ row: 1, column: 1, rowCount: values.length - 1, columnCount: values[0].length - 1 }).assign({ textStyle: { typeface: font, fontSize: 14, color: C.ink } });
  t.columnTracks = columns.map(value => ({ mode: "fr", value }));
  notes(s, note);
  return s;
}
function section(title, subtitle) {
  const s = base(title);
  rect(s, 74, 175, 9, 330, C.teal);
  box(s, subtitle, 116, 210, 900, 160, { fontSize: 33, bold: true, color: C.navy });
  box(s, "Paper-by-paper explanation: problem, model, solution, evidence, and interpretation", 118, 434, 850, 34, { fontSize: 19, color: C.slate });
  return s;
}

// 1
{
  const s = P.slides.add(); no += 1; s.background.fill = C.navy;
  rect(s, 76, 132, 8, 408, C.teal);
  box(s, "Game Theory for Modeling\nPoisoning Attacks in Federated Learning", 122, 156, 990, 170, { fontSize: 45, bold: true, color: C.white });
  box(s, "Concepts, game families, evaluation metrics, and detailed paper studies", 126, 360, 880, 42, { fontSize: 23, color: C.aqua });
  box(s, "Literature review presentation", 126, 456, 420, 28, { fontSize: 17, color: "#B8CAD0" });
  notes(s, "Scope: literature review of game theory for poisoning attacks in federated learning. This deck introduces concepts before analyzing five representative papers.");
}
// 2
{
  const s = base("Strategic motivation");
  box(s, "Poisoning can involve deliberate adaptation: an attacker chooses how to harm learning while a defender chooses how to respond.", 74, 150, 1115, 64, { fontSize: 25, bold: true, color: C.navy });
  item(s, "Attacker choice", "Target, poisoning ratio, trigger, timing, or how closely a malicious update resembles a benign one.", 98, 282, 490, C.gold);
  item(s, "Defender choice", "Admission, weighting, filtering, robust aggregation, post-training repair, or local isolation.", 650, 282, 490, C.teal);
  item(s, "Strategic interaction", "Each side anticipates how the other side may react. The payoff therefore depends on both decisions.", 98, 450, 1040, C.aqua);
  notes(s, "Motivation synthesized from Sagduyu, Erpek, and Shi (2023), arXiv:2312.17164; Jia et al. (2023), FedGame, NeurIPS 2023; Li et al. (2024), arXiv:2410.17431.");
}
// 3
{
  const s = base("Presentation roadmap");
  item(s, "I. Foundations", "Game-theory vocabulary, representations, equilibria, game families, and evaluation metrics.", 100, 176, 1040, C.teal);
  item(s, "II. Detailed case studies", "Five papers with their research question, strategic model, solution method, experimental evidence, and boundaries.", 100, 330, 1040, C.gold);
  item(s, "III. Comparison", "Which choices make the papers different, and which modeling ingredients matter for later research design.", 100, 484, 1040, C.aqua);
  notes(s, "Roadmap for the literature review.");
}
// 4
{
  const s = base("What game theory contributes");
  box(s, "Game theory studies decision-making when the outcome for one actor depends on actions chosen by other actors.", 74, 148, 1100, 56, { fontSize: 25, bold: true, color: C.navy });
  keyValue(s, "Descriptive use", "Explain what rational actors may do under stated incentives and information.", 98, 270, 430, C.teal);
  keyValue(s, "Analytical use", "Derive equilibrium, best response, or a bound that connects choices with outcomes.", 690, 270, 430, C.gold);
  keyValue(s, "Design use", "Construct a defense policy that accounts for an adaptive attacker instead of a fixed attack.", 98, 440, 1020, C.aqua);
  notes(s, "General game-theory framing. Application-specific examples follow the reviewed FL papers.");
}
// 5
tableSlide("Modeling vocabulary", [
  ["Concept", "Meaning", "Poisoning example"],
  ["Player", "Entity that chooses actions", "Attacker, server, receiving node, or client"],
  ["Action", "Feasible choice at a decision point", "Poison ratio; accept, reject, reweight"],
  ["Strategy / policy", "Rule mapping information to an action", "Score an update and select aggregation weight"],
  ["Utility / payoff", "Numerical preference over outcomes", "Clean accuracy, ASR, effort, false-positive cost"],
  ["Type", "Private characteristic that may alter behavior", "Attack family, target, attacker knowledge"],
  ["State", "Information that evolves across rounds", "Current global model and defense history"],
], [1.05, 1.65, 2.25], "Conceptual glossary used throughout the deck. Paper-specific instantiations are cited in the later case studies.");
// 6
{
  const s = base("Players, actions, and strategies");
  box(s, "A game model starts with a precise question: who can choose what, at which time, based on which observations?", 74, 145, 1120, 44, { fontSize: 23, color: C.slate });
  keyValue(s, "Players", "Defender and attacker in adversarial FL; local model and aggregate model in pFedGame.", 94, 244, 330, C.teal);
  keyValue(s, "Action sets", "The feasible decision variables, such as a poisoning ratio r or an aggregation rule Aggr.", 475, 244, 330, C.gold);
  keyValue(s, "Strategy", "A complete contingent rule. In a repeated setting, it maps observed history or state to an action.", 856, 244, 330, C.red);
  box(s, "A paper can call an algorithm “game-theoretic” only after these decision variables and their connection to outcomes are made explicit.", 128, 500, 1000, 48, { fontSize: 21, bold: true, color: C.navy, alignment: "center" });
  notes(s, "Applied examples: Sagduyu et al. use attacker poisoning probabilities and server admission probabilities. FedGame uses attacker poisoning ratio and trigger; Meta Stackelberg uses defense and attack policies.");
}
// 7
{
  const s = base("Utility functions and trade-offs");
  box(s, "The utility function turns a qualitative objective into a quantity that a player seeks to maximize or minimize.", 74, 148, 1110, 48, { fontSize: 24, bold: true, color: C.navy });
  item(s, "Defender utility", "Usually rewards clean-task performance and penalizes poisoning impact, false positives, communication, or computation.", 98, 262, 500, C.teal, 72);
  item(s, "Attacker utility", "Usually rewards damage or backdoor success and can penalize attack effort, detection, or isolation.", 650, 262, 500, C.gold, 72);
  box(s, "Example trade-off: increasing a backdoor poisoning ratio can increase attack strength while making the local model easier to identify.", 110, 480, 1040, 55, { fontSize: 22, color: C.red, bold: true, alignment: "center" });
  notes(s, "FedGame explicitly represents this trade-off in its attacker objective with a genuine score term and a poisoning-ratio term. Jia et al. (2023), Eq. 4.");
}
// 8
{
  const s = base("Information, beliefs, and hidden types");
  item(s, "Complete information", "Both sides know the payoff-relevant model and action sets. This assumption simplifies analysis but rarely holds fully in FL.", 100, 182, 480, C.teal, 78);
  item(s, "Incomplete information", "A defender may not know client identity, attacker objective, poison ratio, trigger, or attack family.", 650, 182, 480, C.gold, 78);
  item(s, "Belief over types", "A Bayesian model represents uncertainty with a distribution over possible attacker types. New observations can guide adaptation.", 100, 402, 1040, C.aqua, 74);
  notes(s, "Li et al. (2024) define attack type using objective, method, and configuration, then model a domain of types with a prior distribution Q. arXiv:2410.17431, Sec. II.");
}
// 9
{
  const s = base("Timing and state transitions");
  box(s, "One-shot and multi-round games answer different questions.", 74, 148, 1110, 40, { fontSize: 25, bold: true, color: C.navy });
  item(s, "Simultaneous move", "Players choose without first observing the other current action. Nash equilibrium is a common solution concept.", 100, 246, 500, C.teal, 70);
  item(s, "Sequential move", "A leader selects a policy and a follower best-responds. Stackelberg modeling requires a justified order of moves.", 650, 246, 500, C.gold, 70);
  item(s, "Markov transition", "The joint actions update a state, such as global parameters. Future payoffs then depend on the updated state.", 100, 462, 1040, C.aqua);
  notes(s, "Sagduyu et al. analyze Nash equilibrium. Li et al. define a Bayesian Stackelberg Markov game with one state transition per FL round.");
}
// 10
tableSlide("Game representations", [
  ["Representation", "Core ingredients", "Use in this review"],
  ["Normal form", "Players, action sets, payoffs", "Compact attack-admission choices"],
  ["Extensive form", "Actions ordered in a decision tree", "Leader-follower reasoning"],
  ["Bayesian game", "Private types and beliefs", "Unknown attacker objective or configuration"],
  ["Markov game", "State, actions, transition, rewards", "FL rounds and changing global model"],
  ["Coalitional value game", "Value of subsets of contributors", "Shapley-inspired contribution aggregation"],
], [1.25, 1.75, 1.95], "General representations connected to reviewed papers. The Shapley row is used only as a contribution-valuation lens, not an attacker-defender equilibrium.", C.paleGold);
// 11
tableSlide("Equilibrium and solution concepts", [
  ["Concept", "Question answered", "Interpretation"],
  ["Best response", "What action maximizes a player utility against a fixed opponent action?", "A local optimal reaction"],
  ["Nash equilibrium", "Can any player improve alone?", "No unilateral incentive to deviate"],
  ["Bayesian Stackelberg equilibrium", "How does a leader choose under uncertainty about follower type?", "Leader anticipates type-specific best responses"],
  ["Meta-Stackelberg equilibrium", "How should a leader prepare to adapt after observing an online trajectory?", "Equilibrium over an initial policy plus adaptation rule"],
  ["Minimax solution", "How can a defender minimize an attacker-favorable objective?", "Adversarial optimization for a zero-sum-like conflict"],
], [1.35, 2.25, 1.35], "Terminology tied to Sagduyu et al. (Nash), Jia et al. (minimax), and Li et al. (Bayesian and meta-Stackelberg).", C.paleTeal);
// 12
tableSlide("Evaluation metrics used in poisoning studies", [
  ["Metric", "Definition", "Why it matters"],
  ["Test accuracy (TA)", "Fraction of clean test inputs classified correctly", "Clean utility of the learned model"],
  ["Attack success rate (ASR)", "Fraction of trigger inputs classified as attacker target", "Backdoor effectiveness; lower is better for a defense"],
  ["Loss / degradation", "Clean or adversarial loss, depending on attack objective", "Untargeted model-poisoning damage"],
  ["Genuine / trust score", "Per-client score used as an aggregation weight", "Separates suspicious from benign client behavior"],
  ["False-positive cost", "Penalty for excluding or downweighting benign contributions", "Defense fairness and utility trade-off"],
  ["Overhead", "Extra computation or communication per round", "Deployability at scale"],
], [1.35, 2.15, 1.45], "FedGame defines TA and ASR explicitly and reports genuine-score computation cost. Meta Stackelberg reports clean global accuracy and backdoor accuracy. Sagduyu et al. use classification accuracy and utilities.", C.paleGold);
// 13
tableSlide("Literature map", [
  ["Paper", "Game-theory mechanism", "Primary question"],
  ["Sagduyu et al. (2023)", "Non-cooperative Nash game", "How many clients to poison or admit?"],
  ["FedGame (2023)", "Minimax attack-defense game", "How can a server resist dynamic backdoors?"],
  ["Meta Stackelberg (2024)", "Bayesian Stackelberg Markov game", "How can defense adapt to uncertain, mixed attacks?"],
  ["pFedGame (2024)", "Two-player constant-sum cooperative weighting", "How should a node mix its own and peer aggregate model?"],
  ["FORCE (2026)", "Shapley-inspired contribution aggregation", "How can DFL identify Byzantine neighbors without gradient checks?"],
], [1.45, 2.0, 1.5], "Sources: Sagduyu et al. (2023), Jia et al. (2023), Li et al. (2024), Behera and Chakraborty (2024), Dong et al. (2026).");

// Sagduyu 14-18
section("Case study 1", "Securing NextG Systems against Poisoning Attacks on Federated Learning\nSagduyu, Erpek, and Shi (2023)");
{
  const s = base("NextG paper: research question and setting");
  item(s, "Application", "Federated signal classification in a NextG communication setting. Each client trains locally and sends model parameters to a server.", 98, 184, 1040, C.teal, 70);
  item(s, "Threat", "An attacker chooses clients whose training labels are poisoned. The server may admit or reject clients from aggregation.", 98, 344, 1040, C.gold, 70);
  item(s, "Central tension", "Rejecting clients can remove poisoned contributions, but rejecting too many also reduces useful data and carries admission cost.", 98, 504, 1040, C.red, 60);
  plain(s, "The paper asks a simple question: how cautious should the server be when caution itself can hurt learning?");
  notes(s, "Sagduyu, Erpek, and Shi, Securing NextG Systems against Poisoning Attacks on Federated Learning: A Game-Theoretic Solution, arXiv:2312.17164, Secs. II-III.");
}
{
  const s = base("NextG paper: the choices in the game");
  keyValue(s, "Attacker action", "For two clients, p_i is the probability of attacking client i. For n clients, m is the number of poisoned clients.", 96, 185, 490, C.gold);
  keyValue(s, "Defender action", "For two clients, q_i is the probability of admitting client i. For n clients, i is the number admitted.", 680, 185, 490, C.teal);
  keyValue(s, "Outcome statistic", "U_{k|i} denotes classification accuracy when i clients are admitted and k of them are poisoned.", 96, 402, 1070, C.aqua);
  plain(s, "Instead of inspecting every neural-network parameter, the model counts two decisions: how many clients are poisoned and how many are admitted.");
  notes(s, "Sagduyu et al. (2023), Secs. IV-V, Eq. 1 and the definitions of p, q, i, m, and U_{k|i}.");
}
{
  const s = base("NextG paper: why each side cares");
  box(s, "Both players value the resulting FL accuracy differently and pay their own action cost.", 74, 148, 1110, 46, { fontSize: 24, bold: true, color: C.navy });
  item(s, "Defender utility U_D(i,m)", "Expected classification reward when i are admitted and some are poisoned, minus i c_D for admitting clients.", 100, 267, 500, C.teal, 72);
  item(s, "Attacker utility U_A(i,m)", "Negative of the FL reward, minus m c_A for poisoning clients. Lower accuracy increases attacker utility.", 650, 267, 500, C.gold, 72);
  box(s, "Interpretation: the model treats admission and poisoning as costly strategic choices, rather than purely technical filters.", 120, 500, 1020, 43, { fontSize: 20, color: C.slate, alignment: "center" });
  plain(s, "The server wants good learning with enough honest clients; the attacker wants poor learning without paying too much to poison clients.");
  notes(s, "Sagduyu et al. (2023), Eqs. 20-21. The attacker utility includes the negative FL reward and poisoning cost; the defender utility includes FL reward and admission cost.");
}
{
  const s = base("NextG paper: what Nash equilibrium means here");
  item(s, "Two-client analysis", "The paper writes expected utilities over attack and admission probabilities. KKT conditions characterize constrained optima for p_i and q_i.", 98, 180, 1040, C.teal, 72);
  item(s, "n-client extension", "It computes best responses B_A(i) = arg max_m U_A(i,m) and B_D(m) = arg max_i U_D(i,m).", 98, 353, 1040, C.gold, 60);
  item(s, "Nash condition", "An equilibrium (i*,m*) lies in the intersection of the defender and attacker best-response correspondences.", 98, 502, 1040, C.aqua, 54);
  plain(s, "At equilibrium, neither side wants to change its own choice if it expects the other side to keep its choice unchanged.");
  notes(s, "Sagduyu et al. (2023), Secs. IV-V, Eqs. 5-19 and Eqs. 22-24. KKT conditions are used for the two-client constrained probabilities; best responses define the n-client Nash equilibrium.");
}
{
  const s = base("NextG paper: findings and modeling lesson");
  item(s, "Reported pattern", "As the total client count increases, their modeled defense utility per client decreases and attacker utility increases because poisoning opportunities grow.", 98, 180, 1040, C.teal, 72);
  item(s, "Cost sensitivity", "Raising attack and defense costs changes equilibrium strategies and utilities. The paper visualizes this dependence.", 98, 347, 1040, C.gold, 62);
  item(s, "Lesson", "A simple Nash model can expose an incentive trade-off, but it abstracts away client identities, rich model updates, and time-varying attacker adaptation.", 98, 500, 1040, C.red, 60);
  plain(s, "This is a clean conceptual model: useful for explaining incentives, but not yet a full simulation of real FL behavior.");
  notes(s, "Sagduyu et al. (2023), Figs. 4-6 and Sec. V. The final sentence is an interpretation of the paper's deliberately compact model.");
}

// FedGame 19-24
section("Case study 2: FedGame (Jia et al., 2023)", "A minimax defense against dynamic backdoor attacks in centralized FL");
{
  const s = base("FedGame: threat model and objective");
  item(s, "Threat", "Compromised FL clients inject a trigger and target label so trigger-embedded inputs are classified as the attacker's chosen class.", 98, 180, 1040, C.gold, 66);
  item(s, "Why a game", "The attacker changes its poisoning ratio or trigger to remain effective; the server changes how it identifies and weights client updates.", 98, 340, 1040, C.teal, 66);
  item(s, "Objective", "Maintain clean test accuracy while lowering attack success rate against dynamic backdoor behavior.", 98, 500, 1040, C.aqua, 54);
  plain(s, "FedGame tries to keep normal predictions correct while making the hidden trigger stop working.");
  notes(s, "Jia et al., FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning, NeurIPS 2023, Abstract and Sec. 4.");
}
{
  const s = base("FedGame: genuine score and weighted aggregation");
  box(s, "FedGame assigns each client a genuine score p_i^t and uses it as a weight in the global update.", 74, 148, 1110, 44, { fontSize: 24, bold: true, color: C.navy });
  keyValue(s, "Reverse-engineer", "Build an auxiliary global model from all local updates, then infer a candidate trigger and target class.", 98, 250, 330, C.gold);
  keyValue(s, "Score", "For each local model, measure how often trigger-embedded clean inputs receive the inferred target label.", 475, 250, 330, C.teal);
  keyValue(s, "Aggregate", "Lower p_i^t for suspicious behavior means a smaller contribution to the global model update.", 852, 250, 330, C.aqua);
  box(s, "p_i^t = 1 - fraction of trigger-embedded server-clean inputs predicted as the inferred target class", 130, 495, 1020, 42, { fontSize: 20, bold: true, color: C.red, alignment: "center" });
  plain(s, "A low score means: this client's model behaves suspiciously when the suspected trigger is added, so trust it less in aggregation.");
  notes(s, "Jia et al. (2023), Sec. 4.1, Eqs. 1-2. The auxiliary model construction and use of trigger reverse engineering are detailed in Sec. 4.2.");
}
{
  const s = base("FedGame: the central trade-off");
  item(s, "Defender variable", "Reverse-engineered trigger δ_re and target class y_re. The defender seeks low scores for malicious clients and high scores for benign clients.", 98, 176, 1040, C.teal, 64);
  item(s, "Attacker variable", "Per-client backdoor ratio r_i^t and trigger δ. A larger r_i^t can improve attack success but tends to reduce genuine score.", 98, 340, 1040, C.gold, 64);
  item(s, "Game objective", "The paper writes a minimax objective that combines score separation with λ times the poisoning ratios of malicious clients.", 98, 500, 1040, C.red, 54);
  plain(s, "The attacker wants a strong backdoor that still looks normal; the defender wants to reveal that tension and downweight the attacker.");
  notes(s, "Jia et al. (2023), Sec. 4.1, Eqs. 3-5. The attacker objective explicitly includes the trade-off term lambda sum_i r_i^t.");
}
{
  const s = base("FedGame: how the defender solves the game");
  item(s, "Unknown malicious set", "The server does not know which selected clients are compromised, so it cannot directly solve the ideal objective.", 98, 178, 1040, C.gold, 62);
  item(s, "Auxiliary global model", "Average all submitted updates into an auxiliary model. Under a backdoor, this model should still expose trigger-target behavior.", 98, 336, 1040, C.teal, 66);
  item(s, "Operational defense", "Apply any compatible trigger reverse-engineering method to the auxiliary model, compute every genuine score, then aggregate with those weights.", 98, 502, 1040, C.aqua, 54);
  plain(s, "The server creates a temporary ‘rough average’ to discover the suspicious pattern before trusting any individual update.");
  notes(s, "Jia et al. (2023), Sec. 4.2. The paper uses Neural Cleanse in its experiments, but the framework is stated to be compatible with other trigger reverse-engineering methods.");
}
{
  const s = base("FedGame: how the attacker approximates a best response");
  item(s, "Private validation set", "A malicious client reserves a portion of local data, for example 10%, to estimate whether its locally trained model still appears genuine.", 98, 175, 1040, C.gold, 64);
  item(s, "Approximate score", "Because the attacker does not know the defender's reverse-engineered trigger, it estimates a score using its own trigger and target.", 98, 339, 1040, C.teal, 64);
  item(s, "Grid search", "It searches r_i^t for the largest estimated genuine score plus λr_i^t, then launches a standard backdoor attack with that ratio.", 98, 503, 1040, C.red, 54);
  plain(s, "The attacker tests several attack strengths and keeps the one that balances damage with stealth.");
  notes(s, "Jia et al. (2023), Sec. 4.3. This is an approximate attacker solution because the attacker does not observe the defender's inferred trigger or scores for benign clients.");
}
{
  const s = base("FedGame: evidence, metrics, and boundary");
  item(s, "Metrics", "Test accuracy measures clean performance; attack success rate measures the fraction of trigger inputs predicted as the target class.", 98, 175, 1040, C.teal, 60);
  item(s, "Evaluation", "The paper evaluates MNIST and CIFAR-10 under scaling, DBA, and Neurotoxin attacks, in IID and non-IID settings.", 98, 335, 1040, C.gold, 62);
  item(s, "Reported limitation", "The defense needs a server clean dataset and trigger reverse engineering. The authors report an average 0.148 seconds per client score on one NVIDIA 2080 Ti GPU.", 98, 498, 1040, C.red, 62);
  plain(s, "The defense is promising only if the server can afford a small clean reference set and the extra scoring work each round.");
  notes(s, "Jia et al. (2023), Secs. 6.1-6.3. Metrics are defined in Sec. 6.1. Computation cost is reported in Sec. 6.2. Do not generalize the measured runtime beyond the stated hardware and setup.");
}

// Meta Stackelberg 25-30
section("Case study 3: Meta Stackelberg Game (Li et al., 2024)", "A Bayesian Stackelberg Markov game with meta-learned online adaptation");
{
  const s = base("Meta Stackelberg: problem and two-stage approach");
  item(s, "Problem", "An FL defender faces unknown, adaptive, or mixed poisoning attacks. A fixed defense policy can fail when the attack type changes.", 98, 176, 1040, C.gold, 66);
  item(s, "Pre-training", "Create a simulated environment from generated data and an attack domain, then train a defense meta-policy over attack types.", 98, 339, 1040, C.teal, 66);
  item(s, "Online adaptation", "Use observed trajectory data from real FL rounds to update the defense policy during deployment.", 98, 502, 1040, C.aqua, 54);
  plain(s, "It learns a starting defense in advance, then adjusts it after seeing how the real attack behaves.");
  notes(s, "Li et al., Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks, arXiv:2410.17431, Fig. 1 and Sec. II.");
}
{
  const s = base("Meta Stackelberg: attack types encode uncertainty");
  keyValue(s, "Objective Ω1", "Untargeted or targeted attack objective.", 96, 185, 300, C.gold);
  keyValue(s, "Method Ω2", "Algorithm that generates the attack policy, such as IPM or DBA.", 490, 185, 300, C.teal);
  keyValue(s, "Configuration Ω3", "Hyperparameters, trigger, target label, and attacker knowledge.", 884, 185, 300, C.aqua);
  box(s, "A joint attack type ξ combines these elements across malicious clients. The defender models known types as an attack domain Ξ with a prior Q(Ξ).", 128, 455, 1020, 64, { fontSize: 22, bold: true, color: C.navy, alignment: "center" });
  plain(s, "‘Attack type’ is simply a structured label for what the attacker wants, how it attacks, and the settings it uses.");
  notes(s, "Li et al. (2024), Sec. II-A(d), definition of attack type and attack domain.");
}
{
  const s = base("Meta Stackelberg: what the game model represents");
  keyValue(s, "State S", "Global model weights w_g^t at the current FL round.", 96, 180, 330, C.teal);
  keyValue(s, "Defender action A_D", "A combination of robust aggregation Aggr(.) and post-training defense h(.).", 475, 180, 330, C.gold);
  keyValue(s, "Attacker action A_ξ", "Joint malicious model updates, selected according to attacker type ξ.", 854, 180, 330, C.red);
  keyValue(s, "Transition T", "The aggregation rule converts current state and joint actions into the next global model.", 96, 400, 500, C.aqua);
  keyValue(s, "Rewards r_D and r_ξ", "Defender seeks lower post-defense loss; attacker seeks the opposing attack objective.", 682, 400, 500, C.teal);
  plain(s, "Each training round becomes one turn of a game: both sides act, the global model changes, and the next turn starts from that new model.");
  notes(s, "Li et al. (2024), Sec. II-A, Bayesian Stackelberg Markov game tuple and bullet definitions. The paper writes the tuple as <S, A_D, A_ξ, T, r, gamma, H>.");
}
{
  const s = base("Meta Stackelberg: why the defender learns to adapt");
  item(s, "Bayesian Stackelberg equilibrium", "The defender chooses a policy that maximizes expected reward over attack types while each type chooses a best response.", 98, 178, 1040, C.teal, 68);
  item(s, "Limitation of fixed BSE", "It is ex-ante: the policy is chosen from the prior before the defender uses new online evidence about the hidden type.", 98, 343, 1040, C.gold, 64);
  item(s, "Meta-Stackelberg equilibrium", "The leader optimizes an initial policy together with a gradient-based adaptation step using a sampled online trajectory.", 98, 500, 1040, C.aqua, 58);
  plain(s, "A fixed plan guesses before training begins. Meta-SE prepares a plan that can learn from evidence observed during training.");
  notes(s, "Li et al. (2024), Sec. III-B, Definitions 1 and 2. The distinction between BSE and meta-SE is the online adaptation of the defender policy.");
}
{
  const s = base("Meta Stackelberg: learning the policy");
  item(s, "Nested optimization", "For sampled attack types, an inner loop learns an attacker best response while the current defender policy remains fixed.", 98, 174, 1040, C.gold, 62);
  item(s, "Outer update", "The returned attacker policy helps estimate the defender value gradient and update the defense meta-policy.", 98, 336, 1040, C.teal, 62);
  item(s, "First-order variant", "To avoid expensive Hessian estimation, the paper introduces Reptile meta-SL, a fully first-order pre-training method.", 98, 498, 1040, C.aqua, 60);
  plain(s, "The algorithm repeatedly lets attackers improve in simulation, then trains the defender to react better next time.");
  notes(s, "Li et al. (2024), Sec. III-C and Algorithm 1. The paper calls its two-timescale policy-gradient method meta-Stackelberg learning and describes a Reptile meta-SL variant.");
}
{
  const s = base("Meta Stackelberg: experiments and interpretation");
  item(s, "Threat coverage", "Untargeted IPM, LMP, and RL attacks; targeted BFL, DBA, and BRL attacks; and mixed attacks are considered.", 98, 174, 1040, C.gold, 62);
  item(s, "Metrics", "The paper reports global-model accuracy and backdoor accuracy. For its Table 2, accuracy is higher and backdoor accuracy is lower.", 98, 336, 1040, C.teal, 62);
  item(s, "Boundary", "The method assumes a simulator, an attack domain, and a way to estimate rewards online. White-box, gray-box, and black-box settings have different practical strength.", 98, 498, 1040, C.red, 62);
  plain(s, "Its strongest results rely on more knowledge about the attack; the realistic setting becomes harder when that knowledge is missing.");
  notes(s, "Li et al. (2024), Sec. IV and Table 2. The practical boundary follows the paper's explicit white-box, gray-box, and black-box evaluations and online reward-estimation discussion.");
}

// pFedGame 31-35
section("Case study 4: pFedGame (Behera and Chakraborty, 2024)", "A DFL game for peer selection and local aggregation weights");
{
  const s = base("pFedGame: purpose and scope");
  item(s, "Setting", "Decentralized FL under dynamic topology. Nodes exchange models directly with immediate neighbors, without a central server.", 98, 176, 1040, C.teal, 64);
  item(s, "Goal", "Improve a node's local model by selecting useful peers and choosing a weight between its own model and peer aggregate.", 98, 339, 1040, C.gold, 64);
  item(s, "Important scope", "The paper does not model a malicious attacker. It is relevant as a DFL game-theoretic aggregation paper, not an adversarial poisoning-defense game.", 98, 502, 1040, C.red, 58);
  plain(s, "pFedGame asks ‘which model should a node trust more?’, not ‘how should a defender fight an attacker?’");
  notes(s, "Behera and Chakraborty, pFedGame - Decentralized Federated Learning using Game Theory in Dynamic Topology, arXiv:2410.04058, Abstract and Secs. I-III.");
}
{
  const s = base("pFedGame: peer selection");
  box(s, "A node x selects neighbor c when the local validation accuracy of c's model meets a threshold θ.", 74, 148, 1110, 44, { fontSize: 23, bold: true, color: C.navy });
  keyValue(s, "Candidate set C(x)", "Immediate neighbors connected to x in the current dynamic graph.", 96, 255, 330, C.teal);
  keyValue(s, "Selection test", "H(M(c), D(x)) ≥ θ: evaluate peer c's model M(c) on x's local data D(x).", 475, 255, 330, C.gold);
  keyValue(s, "Result", "Aggregate only models from selected peers, then compare this peer aggregate with x's own model.", 854, 255, 330, C.aqua);
  plain(s, "A node first asks: does this neighbor’s model work well on my local data? If yes, include it in the peer group.");
  notes(s, "Behera and Chakraborty (2024), Sec. III-A, Eq. 1. H is an accuracy function, M(c) is peer c's model, and D(x) is node x's data.");
}
{
  const s = base("pFedGame: two-player aggregation game");
  item(s, "Players", "Player 1 is the node's own model M(x); player 2 is the aggregate M(α) of selected peer models.", 98, 175, 1040, C.teal, 62);
  item(s, "Actions", "Choose weights ψ(x) and ψ(α) subject to ψ(x) + ψ(α) = 1 in a convex combination of the two models.", 98, 337, 1040, C.gold, 62);
  item(s, "Payoff", "Maximize the accuracy of the weighted combined model on node x's local data. The paper calls this a two-player constant-sum cooperative game.", 98, 499, 1040, C.aqua, 60);
  plain(s, "The two ‘players’ are not people competing. They are two information sources: my model and the neighbors’ combined model.");
  notes(s, "Behera and Chakraborty (2024), Sec. III-B, Eqs. 3-4. The paper's stated terminology is 'two player constant sum cooperative game.'");
}
{
  const s = base("pFedGame: weight-search procedure");
  item(s, "Initialize", "Start with ψ(x)=0 and ψ(α)=1, meaning the peer aggregate initially has full weight.", 98, 174, 1040, C.gold, 58);
  item(s, "Perturb and test", "Increase ψ(x) by δ and decrease ψ(α) by δ. Evaluate the changed combination for r game rounds on local data.", 98, 332, 1040, C.teal, 64);
  item(s, "Accept when useful", "Keep the new weights only if local accuracy improves by more than β. The paper gives time complexity O(|C(x)|)+O(r).", 98, 496, 1040, C.aqua, 60);
  plain(s, "It is a simple trial-and-keep rule: shift a little weight toward the local model, keep the change only when it helps locally.");
  notes(s, "Behera and Chakraborty (2024), Algorithm 1 and Sec. III-C. Parameters delta, beta, and r control the local weight search.");
}
{
  const s = base("pFedGame: results and lesson for DFL");
  item(s, "Evaluation", "Fashion-MNIST, CIFAR-10, and CIFAR-100 under extreme, severe, and homogeneous data heterogeneity, compared with pFedGraph.", 98, 174, 1040, C.teal, 62);
  item(s, "Reported pattern", "It performs comparably in some heterogeneous settings but has lower reported average accuracy than pFedGraph on CIFAR-10 and CIFAR-100 in the paper's summary table.", 98, 336, 1040, C.gold, 70);
  item(s, "Lesson", "DFL can use game vocabulary to formalize a local aggregation trade-off. This differs from modeling attacker and defender incentives under poisoning.", 98, 505, 1040, C.red, 54);
  plain(s, "The value of pFedGame for this review is the DFL aggregation idea, not a direct solution to poisoning attacks.");
  notes(s, "Behera and Chakraborty (2024), Sec. IV and Tables II-III. The interpretation distinguishes pFedGame's aggregation objective from adversarial attack-defense games.");
}

// FORCE 36-40
section("Case study 5: FORCE (Dong et al., 2026)", "Shapley-inspired contribution aggregation for Byzantine-resilient DFL");
{
  const s = base("FORCE: motivation and system setting");
  item(s, "Setting", "Decentralized FL lets clients exchange models without a server. Each node may also act as a local aggregator.", 98, 176, 1040, C.teal, 60);
  item(s, "Threat", "Byzantine clients can send arbitrary or adaptive malicious updates. The paper argues that gradient-checking defenses can be manipulated.", 98, 336, 1040, C.gold, 62);
  item(s, "Goal", "Identify potentially malicious neighbors using a universal contribution measure based on local-model loss rather than a particular gradient pattern.", 98, 500, 1040, C.red, 58);
  plain(s, "Instead of asking ‘does this update look strange?’, FORCE asks ‘does this neighbor actually improve my model?’");
  notes(s, "Dong et al., FORCE: Byzantine-Resilient Decentralized Federated Learning via Game-Theoretic Contribution Aggregation, IEEE TIFS 21 (2026), DOI:10.1109/TIFS.2026.3674690. Public abstract and publication metadata.");
}
{
  const s = base("FORCE: the Shapley-value intuition");
  box(s, "For a set of contributors, the Shapley value assigns each contributor its average marginal contribution across possible coalition orders.", 74, 148, 1110, 50, { fontSize: 24, bold: true, color: C.navy });
  item(s, "Coalition value", "Define how good an aggregate is. A contributor gains value when its addition improves that coalition outcome.", 100, 284, 500, C.teal, 70);
  item(s, "Fair averaging", "Average marginal contribution over different subsets and orders to avoid crediting only one arbitrary arrival order.", 650, 284, 500, C.gold, 70);
  box(s, "In FORCE, the published abstract says the contribution signal comes from local-model loss and helps identify potentially malicious clients.", 114, 512, 1050, 42, { fontSize: 20, color: C.red, bold: true, alignment: "center" });
  plain(s, "A helpful neighbor should improve a coalition. A harmful one should receive little or negative contribution credit.");
  notes(s, "General Shapley-value intuition. FORCE-specific claim derives from Dong et al. (2026) public abstract, DOI:10.1109/TIFS.2026.3674690.");
}
{
  const s = base("FORCE: published method description");
  item(s, "Contribution signal", "FORCE draws inspiration from Shapley value and evaluates a local-model loss independent of specific gradients.", 98, 178, 1040, C.teal, 62);
  item(s, "Aggregation use", "This contribution evaluation supports identification of potentially malicious clients before or during local aggregation.", 98, 340, 1040, C.gold, 62);
  item(s, "Game-theory role", "The game-theory component is contribution valuation, rather than a reported Nash, minimax, or Stackelberg attacker-defender equilibrium.", 98, 502, 1040, C.red, 58);
  plain(s, "Here, game theory supplies a way to share credit fairly. It does not model two strategic opponents choosing actions.");
  notes(s, "Dong et al. (2026), public abstract via IEEE-linked publication record. This slide limits claims to the published abstract description; it does not infer undisclosed algorithmic details.");
}
{
  const s = base("FORCE-: scalability through approximation");
  item(s, "Cost problem", "Exact Shapley-style calculations can grow expensive as a node has more neighboring clients.", 98, 182, 1040, C.gold, 60);
  item(s, "FORCE-", "The paper introduces FORCE-, described as a computation-lightweight variant that approximates Shapley-value computation.", 98, 344, 1040, C.teal, 62);
  item(s, "Deployment implication", "Approximation matters in DFL because local aggregators can have limited compute resources and decentralized neighbor counts vary.", 98, 506, 1040, C.aqua, 56);
  plain(s, "FORCE- sacrifices an exact contribution calculation so a real decentralized node can run the method with many neighbors.");
  notes(s, "Dong et al. (2026), public abstract. The abstract explicitly states that FORCE- approximates Shapley computation to lower overhead as neighboring clients scale.");
}
{
  const s = base("FORCE: evidence boundary and relevance");
  item(s, "Reported evaluation", "The public abstract reports four datasets and three attack settings, with FORCE outperforming compared Byzantine-resilient DFL aggregation methods.", 98, 174, 1040, C.teal, 64);
  item(s, "Evidence boundary", "The accessible source for this review provides abstract-level method details. Exact scoring equation, approximation procedure, and experimental protocol require the full article before reproducing results.", 98, 340, 1040, C.gold, 74);
  item(s, "Relevance", "FORCE is the closest reviewed paper to DFL plus poisoning resilience, but its game-theory mechanism is contribution aggregation rather than strategic attacker-defender equilibrium.", 98, 514, 1040, C.red, 56);
  plain(s, "It is highly relevant to DFL resilience, but it answers a different game-theory question from FedGame or Meta Stackelberg.");
  notes(s, "Dong et al. (2026), public abstract and bibliographic record. The boundary is stated explicitly to prevent unsupported reconstruction of the closed full text.");
}

// Comparison and conclusion
tableSlide("Cross-paper comparison", [
  ["Paper", "Players / choices", "Solution", "Main metric focus"],
  ["Sagduyu et al.", "Attacker poisons; server admits", "Nash best responses", "Classification reward and action costs"],
  ["FedGame", "Attacker selects ratio and trigger; server scores updates", "Minimax plus operational heuristics", "TA, ASR, score cost"],
  ["Meta Stackelberg", "Defender policy; type-specific adaptive attacker", "BSE, meta-SE, policy gradient", "Global accuracy, backdoor accuracy"],
  ["pFedGame", "Own model versus peer aggregate", "Local weight search", "Local model accuracy"],
  ["FORCE", "Neighbor contributions in DFL", "Shapley-inspired aggregation", "Byzantine robustness and overhead"],
], [1.15, 1.9, 1.65, 1.35], "Synthesis based on the methods described in the cited papers. FORCE details remain restricted to public abstract-level evidence.", C.paleGold);
{
  const s = base("What changes across the papers");
  item(s, "Who acts", "Some papers model an explicit attacker and defender. Others use game theory to value contributions or choose aggregation weights.", 98, 180, 1040, C.teal, 62);
  item(s, "What is uncertain", "From no hidden type in a compact Nash model to unknown attack types and multi-round evidence in Meta Stackelberg.", 98, 341, 1040, C.gold, 62);
  item(s, "What counts as a solution", "Equilibrium analysis, minimax optimization, policy learning, and Shapley-inspired scoring answer distinct research questions.", 98, 502, 1040, C.aqua, 56);
  notes(s, "Synthesis based on the reviewed papers. It avoids treating all uses of game theory in FL and DFL as the same modeling approach.");
}
{
  const s = base("Reading checklist for a game-theoretic FL paper");
  item(s, "1. Identify the players and action sets", "Do the paper's real decision variables correspond to the claimed players?", 98, 160, 1040, C.teal, 48);
  item(s, "2. Inspect information and timing", "Which variables are hidden, and do simultaneous or leader-follower moves reflect the system?", 98, 272, 1040, C.gold, 48);
  item(s, "3. Read the utility or objective", "Which trade-offs are rewarded or penalized? Are clean utility and attack damage both measured?", 98, 384, 1040, C.aqua, 48);
  item(s, "4. Separate theory from implementation", "Does the operational algorithm actually approximate the stated equilibrium or optimization problem?", 98, 496, 1040, C.red, 48);
  notes(s, "Methodological checklist synthesized for literature review. It is used to compare the case studies in this deck.");
}
{
  const s = base("Key takeaways");
  item(s, "Game theory supplies a modeling language", "Players, actions, utilities, information, timing, and solution concepts make strategic assumptions inspectable.", 98, 175, 1040, C.teal, 62);
  item(s, "Adaptive attacks motivate dynamic models", "FedGame and Meta Stackelberg explicitly account for the attacker changing behavior in response to defense.", 98, 337, 1040, C.gold, 62);
  item(s, "DFL uses game theory in more than one way", "pFedGame and FORCE show aggregation and contribution formulations; they should not be presented as identical to an attacker-defender strategic game.", 98, 499, 1040, C.red, 60);
  notes(s, "Synthesis of the reviewed literature. This literature review intentionally does not select a final thesis formulation.");
}
// References
{
  const s = base("Selected references");
  box(s, "1. Sagduyu, Erpek, and Shi. Securing NextG Systems against Poisoning Attacks on Federated Learning: A Game-Theoretic Solution. arXiv:2312.17164, 2023.", 84, 150, 1110, 52, { fontSize: 15, color: C.ink });
  box(s, "2. Jia et al. FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning. NeurIPS 2023.", 84, 232, 1110, 44, { fontSize: 15, color: C.ink });
  box(s, "3. Li et al. Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks. arXiv:2410.17431, 2024.", 84, 310, 1110, 52, { fontSize: 15, color: C.ink });
  box(s, "4. Behera and Chakraborty. pFedGame: Decentralized Federated Learning using Game Theory in Dynamic Topology. arXiv:2410.04058, 2024.", 84, 390, 1110, 52, { fontSize: 15, color: C.ink });
  box(s, "5. Dong et al. FORCE: Byzantine-Resilient Decentralized Federated Learning via Game-Theoretic Contribution Aggregation. IEEE TIFS 21, 2026. DOI: 10.1109/TIFS.2026.3674690.", 84, 470, 1110, 56, { fontSize: 15, color: C.ink });
  box(s, "Research notes: game_theory_research_note.md", 84, 618, 780, 24, { fontSize: 15, color: C.teal, italic: true });
  notes(s, "Bibliographic references. The FORCE paper is included from public bibliographic metadata and abstract; confirm full-text details before formal citation or reproduction.");
}

const candidatePath = path.join(workspaceDir, ".codex-finalizer", "game-theory-poisoning-fl-literature-review-expanded-v2.candidate.pptx");
await (await PresentationFile.exportPptx(P)).save(candidatePath);
const tableOwners = [5, 10, 11, 12, 13, 46];
const requirements = { explicitTotalSlideCount: no, requiredNativeTableOwnerSlides: tableOwners, requiredNativeChartOwnerSlides: [] };
const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", ...tableOwners.flatMap(n => ["--require-native-table-slide", String(n)])],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(workspaceDir, ".codex-finalizer", "game-theory-poisoning-fl-literature-review-expanded-v2.validation.json"),
});
console.log(JSON.stringify({ slides: no, finalPath: FINAL_PPTX, result }, null, 2));
