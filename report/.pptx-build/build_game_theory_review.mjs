import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/home/milo/workspace/project/dacn";
const SKILL_DIR = "/home/milo/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".pptx-build");
const RUNTIME_PYTHON = "/home/milo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const FINAL_PPTX = path.join(workspaceDir, "output/pptx/game-theory-poisoning-fl-literature-review-v2.pptx");
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const font = "Fira Sans";
const P = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  navy: "#0D1B2A", ink: "#122033", slate: "#506174", teal: "#007C83",
  aqua: "#5CC8C8", gold: "#D6A84B", mist: "#F2F6F7", white: "#FFFFFF",
  line: "#C7D4D8", paleTeal: "#E8F5F5", paleGold: "#FBF4E5", red: "#A64646",
};

function box(slide, text, left, top, width, height, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", position: { left, top, width, height }, fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: font, fontSize: 20, color: C.ink, autoFit: "shrinkText",
    marginLeft: 0, marginRight: 0, marginTop: 0, marginBottom: 0,
    ...style,
  };
  return shape;
}

function baseSlide(title, number) {
  const slide = P.slides.add();
  slide.background.fill = C.white;
  box(slide, title, 74, 46, 1020, 52, { fontSize: 33, bold: true, color: C.navy });
  const accent = slide.shapes.add({
    geometry: "rect", position: { left: 74, top: 112, width: 78, height: 6 },
    fill: C.teal, line: { fill: C.teal, width: 0 },
  });
  accent.name = "title accent";
  box(slide, String(number).padStart(2, "0"), 1140, 54, 64, 30, { fontSize: 14, color: C.teal, alignment: "right", bold: true });
  return slide;
}

function addNotes(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

function bullet(slide, title, body, left, top, width, accent = C.teal) {
  const mark = slide.shapes.add({
    geometry: "rect", position: { left, top: top + 7, width: 10, height: 10 },
    fill: accent, line: { fill: accent, width: 0 },
  });
  mark.name = "bullet marker";
  box(slide, title, left + 28, top, width - 28, 30, { fontSize: 21, bold: true, color: C.navy });
  box(slide, body, left + 28, top + 33, width - 28, 58, { fontSize: 17, color: C.slate, breakLine: false });
}

// 1. Cover
{
  const s = P.slides.add();
  s.background.fill = C.navy;
  const edge = s.shapes.add({ geometry: "rect", position: { left: 76, top: 132, width: 8, height: 408 }, fill: C.teal, line: { fill: C.teal, width: 0 } });
  edge.name = "cover accent";
  box(s, "Game Theory for Modeling\nPoisoning Attacks in Federated Learning", 122, 156, 980, 170, { fontSize: 45, bold: true, color: C.white, autoFit: "shrinkText" });
  box(s, "Concepts, game families, and literature review", 126, 360, 760, 42, { fontSize: 24, color: C.aqua });
  box(s, "Academic presentation", 126, 456, 320, 28, { fontSize: 17, color: "#B8CAD0" });
  box(s, "Federated Learning Security", 126, 574, 420, 24, { fontSize: 14, color: C.gold, bold: true });
  addNotes(s, "Scope: game-theory literature review for poisoning attacks in federated learning. Sources appear on the relevant research slides.");
}

// 2. Motivation
{
  const s = baseSlide("Why poisoning can be modeled strategically", 2);
  box(s, "A poisoned update is not only an anomaly. It can be a deliberate response to the defender's expected behavior.", 74, 152, 1120, 64, { fontSize: 25, bold: true, color: C.navy });
  bullet(s, "Attacker choice", "The attacker can choose timing, target, intensity, or whether to mimic benign updates.", 98, 272, 490, C.gold);
  bullet(s, "Defender choice", "The defender can admit, reject, reweight, inspect, or isolate an update or client.", 650, 272, 490, C.teal);
  bullet(s, "Mutual adaptation", "Each decision changes the other side's future incentives, costs, and expected outcomes.", 98, 438, 1040, C.aqua);
  addNotes(s, "Conceptual motivation. Relevant direct strategic-defense sources: Sagduyu, Erpek, and Shi (2023), arXiv:2312.17164; Jia et al. (2023), FedGame, NeurIPS 2023.");
}

// 3. Core vocabulary
{
  const s = baseSlide("Core vocabulary of game-theoretic modeling", 3);
  const table = s.tables.add({
    rows: 6, columns: 3, left: 74, top: 150, width: 1130, height: 448,
    values: [
      ["Concept", "Meaning", "Poisoning example"],
      ["Player", "An entity that chooses actions to improve its utility", "Attacker, defender, client, receiving node"],
      ["Action", "A feasible choice at a decision point", "Attack intensity; accept, reject, or reweight"],
      ["Strategy", "A rule for selecting actions in possible situations", "A policy maps observed updates to a defense action"],
      ["Utility / payoff", "Numerical preference over outcomes", "Model performance, attack success, effort, false-positive cost"],
      ["Information", "What a player knows when acting", "The defender may not know attacker identity or type"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({ fill: C.navy, textStyle: { typeface: font, fontSize: 18, color: C.white, bold: true } });
  table.cells.block({ row: 1, column: 0, rowCount: 5, columnCount: 1 }).assign({ fill: C.paleTeal, textStyle: { typeface: font, fontSize: 17, color: C.navy, bold: true } });
  table.cells.block({ row: 1, column: 1, rowCount: 5, columnCount: 2 }).assign({ textStyle: { typeface: font, fontSize: 16, color: C.ink } });
  table.columnTracks = [{ mode: "fr", value: 1.05 }, { mode: "fr", value: 1.7 }, { mode: "fr", value: 2.2 }];
  addNotes(s, "General game-theory vocabulary synthesized for the presentation. See also Sagduyu et al. (2023), arXiv:2312.17164, and Li et al. (2024), arXiv:2410.17431.");
}

// 4. Game model components
{
  const s = baseSlide("Elements of a game model", 4);
  box(s, "A paper must define these elements before it can claim a strategic game model.", 74, 147, 1020, 34, { fontSize: 21, color: C.slate });
  bullet(s, "Players and action sets", "Who makes decisions, and which actions are feasible for each player.", 100, 228, 480, C.teal);
  bullet(s, "Information and beliefs", "Which observations are public, private, or uncertain at the time of action.", 660, 228, 480, C.gold);
  bullet(s, "Utilities and costs", "How performance, damage, effort, false positives, and overhead affect preferences.", 100, 404, 480, C.aqua);
  bullet(s, "Timing and transition", "Whether choices are simultaneous or sequential, and how one round changes the next state.", 660, 404, 480, C.red);
  addNotes(s, "Conceptual framework for reading the papers. A game model must make entities, actions, information, payoffs, and timing explicit.");
}

// 5. Equilibrium
{
  const s = baseSlide("Equilibrium and solution concepts", 5);
  box(s, "An equilibrium predicts stable behavior within stated assumptions. It does not guarantee safety in every real system.", 74, 148, 1115, 46, { fontSize: 21, color: C.slate, italic: true });
  bullet(s, "Nash equilibrium", "No player improves its utility by changing strategy alone while others remain fixed.", 100, 248, 1000, C.teal);
  bullet(s, "Bayesian Nash equilibrium", "Each player acts under beliefs about hidden types, such as benign or malicious clients.", 100, 354, 1000, C.gold);
  bullet(s, "Stackelberg equilibrium", "A leader commits to a strategy and a follower best-responds. The timing must be justified.", 100, 460, 1000, C.aqua);
  addNotes(s, "Definitions are standard game-theory terminology. The application to FL follows Sagduyu et al. (2023) for Nash reasoning and Li et al. (2024) for Bayesian Stackelberg reasoning.");
}

// 6. Families
{
  const s = baseSlide("Game families used in poisoning research", 6);
  const table = s.tables.add({
    rows: 7, columns: 3, left: 74, top: 146, width: 1130, height: 470,
    values: [
      ["Game family", "Key assumption", "What it captures"],
      ["Normal-form / Nash", "One-stage, simultaneous choices", "A compact attack-defense decision"],
      ["Repeated game", "The interaction continues over rounds", "Trust building, delayed attacks, on-off behavior"],
      ["Bayesian game", "A player type is hidden", "Uncertain attacker identity or attack family"],
      ["Stochastic / Markov game", "State changes after actions", "Adaptation across learning rounds"],
      ["Stackelberg game", "Leader commits; follower responds", "Policy commitment and adaptive best response"],
      ["Coalitional / Shapley", "Coalition contribution is valued", "Incentives or contribution-aware aggregation"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({ fill: C.navy, textStyle: { typeface: font, fontSize: 17, color: C.white, bold: true } });
  table.cells.block({ row: 1, column: 0, rowCount: 6, columnCount: 1 }).assign({ fill: C.paleGold, textStyle: { typeface: font, fontSize: 16, color: C.navy, bold: true } });
  table.cells.block({ row: 1, column: 1, rowCount: 6, columnCount: 2 }).assign({ textStyle: { typeface: font, fontSize: 15, color: C.ink } });
  table.columnTracks = [{ mode: "fr", value: 1.2 }, { mode: "fr", value: 1.6 }, { mode: "fr", value: 2.0 }];
  addNotes(s, "Game-family classification synthesized from the reviewed papers. Direct sources: Sagduyu et al. (2023); Jia et al. (2023); Li et al. (2024); Behera and Chakraborty (2024); Dong et al. (2026).");
}

// 7. Information and dynamics
{
  const s = baseSlide("Information and dynamics across training rounds", 7);
  box(s, "Incomplete information", 94, 182, 420, 34, { fontSize: 24, bold: true, color: C.navy });
  box(s, "A defender cannot directly observe whether an unusual update comes from malicious behavior or benign data heterogeneity.", 94, 232, 420, 126, { fontSize: 20, color: C.slate });
  box(s, "Repeated interaction", 738, 182, 420, 34, { fontSize: 24, bold: true, color: C.navy });
  box(s, "Training rounds create history. Previous actions can change trust, detection risk, model state, and later incentives.", 738, 232, 420, 126, { fontSize: 20, color: C.slate });
  box(s, "Bayesian models represent uncertainty through beliefs over hidden types. Markov models represent how the state changes after actions.", 184, 470, 900, 62, { fontSize: 23, bold: true, color: C.teal, alignment: "center" });
  addNotes(s, "Li et al. (2024), Meta Stackelberg Game, models uncertain attack types with a Bayesian Stackelberg Markov game and online adaptation. arXiv:2410.17431.");
}

// 8. Literature map
{
  const s = baseSlide("Literature map: five different uses of game theory", 8);
  const table = s.tables.add({
    rows: 6, columns: 3, left: 74, top: 145, width: 1130, height: 472,
    values: [
      ["Paper", "Game-theory approach", "Primary purpose"],
      ["Sagduyu et al. (2023)", "Nash game", "Poisoning and client-admission decisions"],
      ["FedGame (2023)", "Minimax game", "Defense against dynamic backdoor attackers"],
      ["Meta Stackelberg (2024)", "Bayesian Stackelberg Markov game", "Adaptive defense under uncertain attack types"],
      ["pFedGame (2024)", "Constant-sum cooperative weighting", "Localized DFL aggregation in dynamic topology"],
      ["FORCE (2026)", "Shapley-inspired contribution aggregation", "Byzantine-resilient DFL aggregation"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({ fill: C.navy, textStyle: { typeface: font, fontSize: 17, color: C.white, bold: true } });
  table.cells.block({ row: 1, column: 0, rowCount: 5, columnCount: 1 }).assign({ fill: C.paleTeal, textStyle: { typeface: font, fontSize: 16, color: C.navy, bold: true } });
  table.cells.block({ row: 1, column: 1, rowCount: 5, columnCount: 2 }).assign({ textStyle: { typeface: font, fontSize: 15, color: C.ink } });
  table.columnTracks = [{ mode: "fr", value: 1.25 }, { mode: "fr", value: 1.8 }, { mode: "fr", value: 1.9 }];
  addNotes(s, "Sources: Sagduyu, Erpek, and Shi (2023), arXiv:2312.17164; Jia et al. (2023), FedGame, NeurIPS 2023; Li et al. (2024), arXiv:2410.17431; Behera and Chakraborty (2024), arXiv:2410.04058; Dong et al. (2026), IEEE TIFS, DOI 10.1109/TIFS.2026.3674690.");
}

// 9. Paper comparison
{
  const s = baseSlide("How the reviewed papers differ", 9);
  const table = s.tables.add({
    rows: 6, columns: 4, left: 60, top: 145, width: 1160, height: 485,
    values: [
      ["Paper", "Players", "Information / timing", "Strategic focus"],
      ["Sagduyu et al.", "Attacker and defender", "Compact poisoning-admission game", "Nash utility trade-off"],
      ["FedGame", "Dynamic attacker and defender", "Adaptive interaction", "Minimax robustness against backdoors"],
      ["Meta Stackelberg", "Defender leader; attacker follower", "Hidden attack type; multi-round state", "Meta-policy adaptation"],
      ["pFedGame", "Own model and aggregate model", "Local aggregation rounds", "Weight optimization, not adversarial defense"],
      ["FORCE", "Contributors / local aggregation", "Contribution evaluation", "Shapley-inspired Byzantine resilience"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({ fill: C.navy, textStyle: { typeface: font, fontSize: 16, color: C.white, bold: true } });
  table.cells.block({ row: 1, column: 0, rowCount: 5, columnCount: 1 }).assign({ fill: C.paleGold, textStyle: { typeface: font, fontSize: 15, color: C.navy, bold: true } });
  table.cells.block({ row: 1, column: 1, rowCount: 5, columnCount: 3 }).assign({ textStyle: { typeface: font, fontSize: 14, color: C.ink } });
  table.columnTracks = [{ mode: "fr", value: 1.0 }, { mode: "fr", value: 1.45 }, { mode: "fr", value: 1.6 }, { mode: "fr", value: 1.75 }];
  addNotes(s, "Comparison based on the papers' stated models and evaluated objectives. Sources: Sagduyu et al. (2023), arXiv:2312.17164; Jia et al. (2023), NeurIPS 2023; Li et al. (2024), arXiv:2410.17431; Behera and Chakraborty (2024), arXiv:2410.04058; Dong et al. (2026), DOI 10.1109/TIFS.2026.3674690.");
}

// 10. Synthesis
{
  const s = baseSlide("Literature synthesis", 10);
  bullet(s, "Strategic defense has strong centralized-FL precedents", "Nash, minimax, and Bayesian Stackelberg approaches explicitly model attacker and defender incentives.", 98, 180, 1040, C.teal);
  bullet(s, "DFL work uses game theory in a different way", "pFedGame focuses on local aggregation weights, while FORCE uses contribution valuation for Byzantine resilience.", 98, 330, 1040, C.gold);
  bullet(s, "The terminology needs care", "A game-theoretic aggregation heuristic differs from an adversarial strategic game with players, actions, information, payoffs, and equilibrium.", 98, 480, 1040, C.red);
  addNotes(s, "Synthesis from all reviewed sources. The distinctions reflect the papers' stated players, objectives, and evaluation settings rather than a claim of complete literature coverage.");
}

// 11. References
{
  const s = baseSlide("Selected references", 11);
  box(s, "1. Sagduyu, Erpek, and Shi. Securing NextG Systems against Poisoning Attacks on Federated Learning: A Game-Theoretic Solution. arXiv:2312.17164, 2023.", 84, 160, 1110, 54, { fontSize: 16, color: C.ink });
  box(s, "2. Jia et al. FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning. NeurIPS 2023.", 84, 244, 1110, 48, { fontSize: 16, color: C.ink });
  box(s, "3. Li et al. Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks. arXiv:2410.17431, 2024.", 84, 326, 1110, 54, { fontSize: 16, color: C.ink });
  box(s, "4. Behera and Chakraborty. pFedGame: Decentralized Federated Learning using Game Theory in Dynamic Topology. arXiv:2410.04058, 2024.", 84, 414, 1110, 54, { fontSize: 16, color: C.ink });
  box(s, "5. Dong et al. FORCE: Byzantine-Resilient Decentralized Federated Learning via Game-Theoretic Contribution Aggregation. IEEE TIFS, 2026. DOI: 10.1109/TIFS.2026.3674690.", 84, 500, 1110, 54, { fontSize: 16, color: C.ink });
  box(s, "Full links and reading notes: game_theory_research_note.md", 84, 620, 820, 24, { fontSize: 15, color: C.teal, italic: true });
  addNotes(s, "Bibliographic references are included visibly on this final slide. Verify final publication status for arXiv preprints before using them in the formal thesis bibliography.");
}

const candidatePath = path.join(workspaceDir, ".codex-finalizer", "game-theory-poisoning-fl-literature-review-v2.candidate.pptx");
await (await PresentationFile.exportPptx(P)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 11,
  requiredNativeTableOwnerSlides: [3, 6, 8, 9],
  requiredNativeChartOwnerSlides: [],
};
const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "3", "--require-native-table-slide", "6", "--require-native-table-slide", "8", "--require-native-table-slide", "9"],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(workspaceDir, ".codex-finalizer", "game-theory-poisoning-fl-literature-review-v2.validation.json"),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, result }, null, 2));
