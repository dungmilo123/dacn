# Đặc tả khởi đầu: simulator DFL và Bayesian best-response defense

> Tổng hợp quyết định và giới hạn từ phiên thảo luận thiết kế.
> Tài liệu này là điểm bắt đầu implementation, không phải kết quả thực nghiệm hoặc chứng minh equilibrium.
> Khi các ghi chú trước đề xuất repeated/adaptive/Markov ngay từ đầu, dùng scope thu gọn trong tài liệu này cho phiên bản implementation đầu tiên. Không thay đổi nội dung các ghi chú trước.

## 1. Đề tài, câu hỏi và sản phẩm cuối

**Đề tài:** Chống tấn công đầu độc mô hình trong Học Liên Kết Phi tập trung dựa trên mô hình trò chơi chiến lược.

**Câu hỏi cho phiên bản đầu:**

> Một benign receiving node nên chọn aggregation weights thế nào khi không biết type của neighbors, dưới một mô hình hành vi model-poisoning attacker đã xác định?

Đóng góp hướng tới là **local defense algorithm suy ra từ Bayesian game formulation**, không chỉ simulator hoặc bảng metrics.

Pipeline:

```text
Simulator DFL nhỏ chạy đúng
    → Model-poisoning actions hữu hạn
    → Logs và metrics đáng tin
    → Bayesian model: types, prior, likelihood, utilities
    → Bayesian best-response decision rule
    → Local aggregation policy
    → Kiểm chứng derivation, solution quality và defense performance
```

Sản phẩm cần có:

1. Threat model và game formulation cụ thể.
2. Procedure calibration/derivation đủ tái hiện.
3. Defense policy với input/output, feasible actions và tie-breaking rõ ràng.
4. Online execution tại mỗi benign node, chỉ dùng local information.
5. Implementation, tests, baseline comparisons và phân tích giới hạn.

Simulator là hạ tầng để tạo và kiểm tra thuật toán. Không bắt buộc node chạy equilibrium solver mỗi round; policy có thể được giải offline rồi compile thành bảng tra hoặc công thức.

## 2. Scope đã chốt cho bản đầu

| Thành phần               | Scope                                                                      |
| ------------------------ | -------------------------------------------------------------------------- |
| Kiểu attack              | Untargeted **model poisoning**, không phải data poisoning                  |
| Attacker controller      | Một controller                                                             |
| Compromised nodes        | Tối đa một node trong một episode; có cả clean episodes                    |
| Graph                    | Nhỏ, cố định trong episode                                                 |
| Communication            | Synchronous rounds                                                         |
| Message protocol         | Trao đổi full model cùng kiến trúc                                         |
| Broadcast                | Một sender gửi cùng một message cho mọi neighbor trong round               |
| Attack family            | Một family ban đầu, vài intensity configurations hữu hạn                   |
| Attack policy            | Configuration được chọn trước episode, giữ cố định qua rounds              |
| Defender                 | Mỗi benign receiving node tự aggregate                                     |
| Defense architecture     | Chung rule/calibration artifact, execution độc lập theo local observations |
| Bayesian information     | Hidden neighbor types, prior và current-evidence likelihood                |
| History/trust memory     | Chưa có trong bản đầu                                                      |
| Sequential belief update | Chưa có trong bản đầu                                                      |
| Solution ban đầu         | Myopic Bayesian best response trong finite local stage-action space        |
| Network equilibrium      | Chưa claim                                                                 |
| Markov/Stackelberg       | Để sau khi hoàn thiện tuyến Bayesian                                       |

Các công thức attack family, finite weight actions và payoff/calibration ở các mục sau là **đề xuất implementation tối giản**, không phải quyết định bất biến hoặc kết quả nghiên cứu. Có thể thay chúng trước experiments, nhưng phải giữ formulation, solver, tests và claims nhất quán. Các numerical hyperparameters chưa chốt được liệt kê ở mục 18.

Không cần bao quát mọi model-poisoning family. Một family đủ để xây và kiểm tra bản đầu; kết luận phải giới hạn tương ứng. Nếu sau này đủ nguồn lực, đánh giá thêm family ngoài calibration để kiểm tra generalization.

**Không thuộc critical path:** recipient-specific messages, nhiều independent attackers, isolation/topology control, warm-up/on–off/adaptive attacks, deep RL, distributed deployment.

## 3. Từ vựng: không đánh đồng các khái niệm

| Khái niệm       | Định nghĩa trong tài liệu                                                      |
| --------------- | ------------------------------------------------------------------------------ |
| Player          | Decision-maker chọn actions để tối ưu utility                                  |
| Action          | Một lựa chọn ở một decision point, ví dụ intensity hoặc một vector weights     |
| Strategy/policy | Luật chọn action trong các tình huống có thể gặp                               |
| Type            | Thông tin riêng về player/node, ban đầu là benign hoặc malicious               |
| Observation     | Thông tin player thực sự nhận được khi quyết định                              |
| Utility/payoff  | Đại lượng biểu diễn điều player muốn tối ưu                                    |
| Feedback        | Tín hiệu player thực sự quan sát sau action; không đồng nghĩa utility          |
| Metric          | Đại lượng evaluator đo để phân tích; không mặc nhiên là policy input           |
| Runtime state   | Dữ liệu simulator cần để vận hành, như local models và optimizer states        |
| Markov state    | Biểu diễn đủ cho transition theo Markov property; chưa đặt làm yêu cầu bản đầu |

Accuracy plot không phải game. Payoff matrix không tự tạo ra defense. Phải có procedure nối game assumptions với policy cuối.

## 4. Players và vai trò của offline designer

### 4.1. Attacker player

Một attacker controller kiểm soát compromised node \(m\). Controller được sửa outgoing model của node đó, không được sửa messages của benign nodes hoặc scheduler.

Controller khác compromised node:

- Một controller quản lý nhiều compromised nodes: có thể là một player chọn joint actions.
- Nhiều attackers độc lập: nhiều players, cần utilities và information structure riêng.
- Hai trường hợp trên không được tự động đánh đồng; đều nằm ngoài scope đầu.

### 4.2. Benign players

Mỗi benign node đồng thời là learner, honest sender và local receiving defender. Node \(i\) chọn weights cho neighborhood của chính nó.

Các nodes dùng chung policy structure/calibration artifact nhưng có observations khác nhau, nên có thể tạo weights khác nhau:

\[
\boldsymbol\alpha_i^t=\pi^*(o_i^t).
\]

Không có online central defender chọn weights thay toàn mạng.

### 4.3. Offline designer/solver

Offline designer là thành phần thiết kế phương pháp, không phải một server trong DFL protocol. Designer có thể dùng privileged simulator logs để calibration.

Nếu chọn policy theo objective chung của benign nodes, đó là **defender-team abstraction ở tầng thiết kế**. Nó không chứng minh Nash equilibrium giữa các benign nodes.

Bản đầu chỉ claim local stage best responses dưới attacker-behavior và môi trường tương tác đã chỉ định. Các local games bị coupled bởi models và graph; không claim nghiệm từng neighborhood ghép thành equilibrium toàn mạng.

## 5. Types và incomplete information

\[
\theta_j\in\{B,M\},
\]

với \(B\) là benign và \(M\) là malicious. Types cố định trong episode. Node benign biết mình không bị controller điều khiển nhưng không biết neighbor types.

Phân biệt:

- **Type:** node có chủ đích tấn công hay không.
- **Action:** round hiện tại gửi honest hay manipulated message.
- **Contamination:** local model đã chịu ảnh hưởng poisoning đến đâu.

Malicious node có thể gửi honest message mà vẫn malicious. Benign node có thể gửi contaminated model mà vẫn benign.

Policy không nên mặc nhiên coi mọi harmful message là từ malicious sender. Likelihood/payoff calibration phải bao gồm trajectories có propagation qua benign intermediaries, không chỉ clean benign messages.

### Episode sampler và prior

Thiết kế một sampler công khai:

1. Với xác suất cấu hình \(p_0\), episode hoàn toàn clean.
2. Nếu attacked, chọn một compromised node từ placement distribution đã định nghĩa.
3. Chọn attack configuration từ distribution dùng cho calibration.

Prior là giả định thiết kế, không phải true identity được truyền cho node. Phải ghi rõ \(p_0\), placement distribution và cách chúng tạo prior cho neighborhood, điều kiện trên receiver là benign.

Do tối đa một compromised node, neighbor types nói chung **không độc lập**. Với graph nhỏ, nên enumerate feasible neighborhood type configurations thay vì nhân các marginal beliefs rồi coi đó là exact joint posterior.

Có thể dùng fixed attacker placement cho debug propagation, nhưng không dùng sự thật “node ID này luôn malicious” làm feature hoặc calibration shortcut của defense.

## 6. Protocol và timeline của simulator

Đặt \(G=(V,E)\), \(N_i\) là tập neighbors gửi đến \(i\), \(R_i^t\subseteq N_i\) là tập có message khả dụng.

Mỗi round thực hiện bằng snapshots:

```text
1. Chụp local models đầu round.
2. Tất cả nodes local-train từ snapshots của mình.
3. Attacker tạo manipulated outgoing message của compromised node.
4. Giao messages theo graph; hoàn tất receive phase.
5. Mỗi benign receiver tạo observation và chọn weights.
6. Aggregate đồng thời để tạo next local models.
7. Evaluator đo, ghi logs; bắt đầu round tiếp theo.
```

Không cập nhật model của node trước rồi cho node khác đọc next model trong cùng synchronous round. Không có global averaging thay local aggregation.

Local training:

\[
\widetilde w_j^t=\operatorname{LocalTrain}(w_j^t,D_j,z_j^t).
\]

Optimizer state \(z_j^t\) được giữ nếu training rule cần. Để giảm complexity ban đầu, có thể chọn optimizer không momentum; đây là configuration, không phải yêu cầu game-theoretic.

Compromised node vẫn có local model/optimizer và thực hiện local training. Đề xuất tối giản: nó tiếp tục bằng own trained model, không có thêm local defense decision trong bản đầu. Quy tắc này phải cố định và được log.

## 7. Attacker actions và strategy

### 7.1. Một model-poisoning family đề xuất để prototype

Đặt local delta:

\[
\Delta_m^t=\widetilde w_m^t-w_m^t.
\]

Outgoing full model:

\[
u_m^t=\widetilde w_m^t-\kappa\Delta_m^t.
\]

- \(\kappa=0\): honest outgoing model.
- \(\kappa=1\): triệt tiêu local training step.
- \(\kappa>1\): đảo chiều local delta.

Chọn finite set \(\mathcal K=\{0,\kappa_{\mathrm{weak}},\kappa_{\mathrm{strong}}\}\), với \(1<\kappa_{\mathrm{weak}}<\kappa_{\mathrm{strong}}\). Giá trị cụ thể là experiment configuration, không phải kết quả suy ra từ game.

Đây là family khởi đầu, không đại diện mọi model poisoning. Kiểm tra thực nghiệm xem nó gây damage rõ trong protocol đã chọn. Không thay nó bằng việc nhân toàn bộ full model với một hệ số rồi coi hai phép manipulation là tương đương.

Mọi recipients nhận cùng \(u_m^t\). Benign sender gửi \(\widetilde w_j^t\), kể cả khi model đã contaminated.

### 7.2. Action khác strategy

- Action tại round \(t\): gửi message với intensity \(\kappa_t\).
- Strategy bản đầu \(\sigma_\kappa\): dùng cùng \(\kappa\) trong mọi rounds của episode.
- Một distribution \(\mu_A\) có thể chọn giữa các strategies trước episode; không phải đổi intensity ngẫu nhiên mỗi round.

Malicious type cũng có thể chọn honest strategy. Honest action không biến malicious type thành benign type.

Bayesian best response được tìm **đối với một specified \(\sigma_A\) hoặc distribution \(\mu_A\)**. Distribution này là assumption/calibration setting, không mặc nhiên là equilibrium strategy.

### 7.3. Attacker information và constraints

Attacker biết own data/model, attack configuration, round index và public defense algorithm. Không tự động được biết private validation sets, receiver posteriors hoặc current receiver weights trước khi gửi.

Bản đầu không dùng feedback để đổi action. Global damage có thể là objective evaluator tính, nhưng không vì vậy trở thành online attacker observation.

Finite intensities tạo initial action bound. Nếu thêm norm cap hoặc attack budget, phải định nghĩa trong configuration và áp dụng nhất quán cho mọi defense.

## 8. Defender action space và online policy

Defender action là **joint vector aggregation weights**, không phải một threshold hay một trust score:

\[
d_i^t=\boldsymbol\alpha_i^t,
\quad \alpha_{ij}^t\ge0,
\quad \sum_{j\in R_i^t\cup\{i\}}\alpha_{ij}^t=1.
\]

### Finite action representation đề xuất

Với reference weights \(b_{ij}>0\), chọn neighbor multipliers:

\[
g_{ij}\in\{0,\omega,1\},\qquad 0<\omega<1.
\]

Self multiplier bằng 1. Đặt:

\[
Z_i=b_{ii}+\sum_{j\in R_i^t}b_{ij}g_{ij},
\qquad
\alpha_{ii}=\frac{b_{ii}}{Z_i},
\qquad
\alpha_{ij}=\frac{b_{ij}g_{ij}}{Z_i}.
\]

- \(g=0\): reject.
- \(g=\omega\): downweight so với reference multiplier.
- \(g=1\): reference/accept multiplier; actual weight vẫn phụ thuộc normalization.
- Reject tất cả hoặc không có messages: self-weight bằng 1.

Candidate actions là các multiplier profiles của **cả neighborhood**. Có tối đa \(3^{|R_i^t|}\) profiles; graph nhỏ/degree nhỏ giúp enumerate. Không tối ưu từng edge độc lập rồi bỏ qua weight normalization.

Policy bản đầu là memoryless theo policy input:

\[
d_i^t=\pi(o_i^t).
\]

Memoryless không có nghĩa model không chứa ảnh hưởng quá khứ; chỉ có nghĩa policy chưa có thêm trust/history memory.

## 9. Local observations, Bayesian beliefs và calibration

### 9.1. Information boundary

Policy được dùng:

- Own current/trained model.
- Received messages và available sender IDs.
- Own local validation data/signals.
- Current round/context được protocol cung cấp.
- Prior và calibration artifact được cấu hình offline.

Policy không được dùng true neighbor types, network-wide loss/accuracy, evaluator hold-out labels, contamination ground truth hoặc toàn bộ graph không được protocol cung cấp.

### 9.2. Current evidence

Đặt \(\delta_{ij}=u_{j\to i}-w_i\) và \(\delta_{ii}=\widetilde w_i-w_i\). Features khởi đầu:

1. Directional disagreement giữa hai deltas.
2. Relative delta norm.
3. \(L_i^{val}(u_{j\to i})-L_i^{val}(\widetilde w_i)\).

Định nghĩa normalization, clipping, zero-norm handling và numerical epsilons trước khi chạy experiments. Low similarity không mặc nhiên malicious, đặc biệt trong non-IID.

Đặt \(x_i\) là finite/discretized evidence và \(c_i\) là observable context nếu sử dụng, ví dụ degree và round-progress bin. Current round không phải history memory.

### 9.3. Posterior bản đầu

Với neighborhood type vector \(\boldsymbol\theta\):

\[
b_i(\boldsymbol\theta\mid x_i,c_i;\mu_A)
\propto
P(x_i\mid\boldsymbol\theta,c_i;\mu_A)
P(\boldsymbol\theta\mid c_i).
\]

Likelihood marginalizes attack choices theo specified behavior model. Ví dụ một sender:

\[
P(x\mid\theta;\mu_A)
=\sum_a P(x\mid\theta,a)\mu_A(a\mid\theta),
\]

với các context cần thiết được hiểu là đã điều kiện hóa.

Đây là inference từ **current evidence**, chưa phải recursive update từ \(b^{t-1}\). Bayesian không bắt buộc có history. Trust score chưa calibration không được gọi là posterior probability.

### 9.4. Calibration path nhỏ, có thể implement

1. Thu clean và attacked calibration trajectories với specified behavior/placement distributions.
2. Lưu local features, context, true types và actual attack actions trong offline-only records.
3. Dùng bins nhỏ để ước lượng likelihood tables; smoothing cho finite-count cells.
4. Thu outcome của candidate defense actions để ước lượng conditional payoff tables.
5. Kiểm tra coverage, posterior calibration và held-out prediction error.
6. Freeze artifact trước final evaluation.

Ghi rõ policy nào tạo calibration trajectories, ví dụ plain averaging và các candidate weighting rules, cùng episode distributions tương ứng. Khi deploy Bayesian policy, kiểm tra coverage/mismatch trên held-out trajectories của policy đó; empirical tables dưới reference policies không tự trở thành exact on-policy model. Nếu cần recollect/recalibrate, chỉ dùng design/selection splits và ghi rõ procedure, không diễn giải vòng calibration này thành equilibrium solver.

Thay discretization bằng classifier/regressor là mở rộng implementation, không phải yêu cầu đầu tiên. Bins, smoothing và fallback phải được versioned; không được âm thầm dùng evaluator labels khi thiếu cell.

Changing attacker strategy, non-IID setting hoặc behavior của other defenders có thể làm likelihood/payoff model thay đổi. Đây là distribution mismatch phải kiểm tra. Models nhiễm poisoning và hidden network effects khiến type beliefs không mặc nhiên là sufficient statistics.

## 10. Utility/payoff, feedback và metrics

### 10.1. Defender payoff cho local stage model

Đề xuất initial payoff: clean task-loss improvement của local candidate aggregate, có thể trừ explicit defense cost:

\[
U_i^D(d)
=L_i^{design}(\widetilde w_i)
-L_i^{design}(w_i'(d))
-\lambda_D C_i(d).
\]

Trong đó:

\[
w_i'(d)=\alpha_{ii}\widetilde w_i
+\sum_{j\in R_i}\alpha_{ij}u_{j\to i}.
\]

- \(D_i^{design}\) là clean held-out data cho offline calibration, tách khỏi node-local validation features và final test data.
- Type không được truyền trực tiếp cho online policy để tính payoff.
- Có thể bắt đầu \(\lambda_D=0\), báo computation overhead riêng. Nếu thêm cost, định nghĩa và đo rõ; reject sau receive không tự tiết kiệm communication của messages đã nhận.
- Không cộng thêm damage penalty trùng với tác động đã nằm trong task loss.

Offline evaluator có thể tạo counterfactual candidate aggregates từ cùng round snapshot để đo payoff, nhưng **chỉ action được policy chọn mới đi vào actual trajectory**. Không thay models/optimizer states của run bằng counterfactual branches.

### 10.2. Vì sao phải ước lượng expected payoff?

Online node không biết exact outcome trên offline design/test population. Policy dùng estimated conditional payoff:

\[
\widehat Q_i(x,c,d,\boldsymbol\theta;\mu_A)
\approx
\mathbb E[U_i^D(d)\mid x,c,\boldsymbol\theta;\mu_A].
\]

Attack choices và các yếu tố chưa quan sát được được tích phân theo calibration model. Không giả định type một mình quyết định message quality.

Nếu utility chỉ là local validation loss mà node có thể tính trực tiếp cho mọi candidate aggregate, và type/posterior không thay đổi decision, đó có thể chỉ là validation-based selection. Phải kiểm tra Bayesian/game component thực sự có vai trò, không thêm nó để đổi tên phương pháp.

### 10.3. Attacker utility

Attacker preference là gây task degradation. Một episode objective đề xuất:

\[
J_A
=\mathbb E\left[
\frac1{|B|}\sum_{i\in B}L_i^{design}(w_i^T)
-\lambda_A\sum_{t=0}^{T-1}\kappa_t^2
\right],
\]

với \(B\) là benign nodes. Có thể đặt \(\lambda_A=0\) cho prototype nếu finite action bound đã được chốt. Đây là objective offline evaluator tính, không phải feedback toàn mạng mà attacker được tự động truy cập.

Initial solver chưa tối ưu \(J_A\); nó dùng specified attacker strategy. Định nghĩa utility giúp attacker là strategic decision-maker của formulation, nhưng không biến strategy đã cho thành best response/equilibrium.

### 10.4. Feedback và evaluation

Local feedback có thể là local validation loss sau aggregation. Bản đầu không dùng feedback để online retrain policy hoặc update historical beliefs.

Metrics tối thiểu:

- Per-node clean task loss/accuracy theo rounds.
- Mean/worst benign-node performance.
- Attack impact so với paired clean runs.
- Weights/rejections và current evidence/posteriors.
- Propagation đến benign nodes không trực tiếp nối attacker.
- Computation overhead và calibration-table fallback frequency.
- Benign-sender rejection và performance under non-IID.

Benign-sender rejection không tự đồng nghĩa quyết định sai: benign message có thể contaminated. Muốn claim harmful-message detection cần một operational ground-truth criterion riêng.

## 11. Policy suy ra từ Bayesian best response

Với posterior và conditional payoff model đã calibration:

\[
\widehat V_i(x,c,d;\mu_A)
=\sum_{\boldsymbol\theta}
b_i(\boldsymbol\theta\mid x,c;\mu_A)
\widehat Q_i(x,c,d,\boldsymbol\theta;\mu_A).
\]

Defense rule:

\[
\boxed{
\pi_i^{BR}(o_i)
=\operatorname{TieBreak}\left(
\arg\max_{d\in\mathcal A_i(R_i)}
\widehat V_i(x_i,c_i,d;\mu_A)
\right)
}
\]

Phải nêu deterministic candidate ordering và tie tolerance. Missing/unseen cells dùng documented calibration-only fallback; nếu hoàn toàn không có supported estimate, có thể fallback local-only và phải log tỷ lệ xảy ra. Fallback là engineering rule, không claim equilibrium/best response ở những cells đó.

Chuỗi derivation:

```text
Types + prior + specified attacker strategy
    → Observation likelihood + conditional payoff model
    → Posterior và expected utility cho feasible actions
    → Argmax với tie-breaking
    → Policy chọn aggregation weights từ local observations
```

Policy không phải một vector weights hoặc “tổng các parameters”. Nó là rule tạo weights cho những observations khác nhau.

Có thể compute expected utilities online hoặc precompute lookup table cho finite observation/context space. Threshold chỉ được gọi là suy ra nếu thực sự có derivation tương ứng; không coi arbitrary threshold là nghiệm game.

**Giới hạn:** đây là approximate/myopic Bayesian best response trong estimated local stage model và finite action space. Không claim optimal full-episode policy, Bayesian Nash equilibrium hoặc network-wide equilibrium.

## 12. Bayesian game, best response và equilibrium

Bayesian là **information structure**. Best response/equilibrium là **solution target**. Chúng không phải các lựa chọn loại trừ nhau.

- Hidden type alone có thể chỉ tạo Bayesian decision problem.
- Strategic game formulation còn cần choices và utilities của attacker/defender.
- Bayesian best-response defense giải một phía đối với specified opponent behavior.
- Bayesian Nash equilibrium tìm strategy profile mà mỗi player/type là best response theo prior và information constraints.

Nếu sau này dùng defender-team abstraction với two-player strategy sets:

\[
\pi_D^_\in\arg\max\_{\pi_D}J_D(\pi_D,\sigma_A^_),
\qquad
\sigma_A^_\in\arg\max\_{\sigma_A}J_A(\pi_D^_,\sigma_A),
\]

với attacker strategy conditional on private type và expectations phù hợp. Đây là illustration của restricted Bayesian strategy-game conditions, không phải proof cho nhiều independent local defenders.

Không được lấy myopic stage solver rồi thay tên thành episode-level best response. Không thay robust max–min bằng chữ Bayesian equilibrium mà không đổi bài toán/solver.

Khi kiểm tra deviation, freeze opposing strategy; không tự retrain bên kia sau mỗi deviation rồi gọi đó là unilateral-deviation test.

## 13. Runtime state có, Markov formulation để sau

Simulator vẫn phải giữ local models, optimizer states, topology, current round, types và attacker configuration. Đây là **runtime bookkeeping**, chưa phải formal global Markov state.

Bản đầu không yêu cầu:

- Exact Markov state hoặc Markov transition kernel.
- Local observation là Markov/sufficient statistic.
- Global model hoặc global-state input cho policy.
- Dynamic programming, Markov equilibrium hay long-horizon optimality.

State definition không tự biến bài toán thành Markov chain. Markov chain thường mô tả stochastic evolution khi policies đã cố định; Markov game còn có players và controlled transitions. Formulating những thứ này là bước sau, không phải prerequisite để viết simulator Bayesian ban đầu.

## 14. Repeated Bayesian và thứ tự mở rộng

DFL chạy nhiều training rounds. Có thể nói hệ thống tạo repeated Bayesian interactions, nhưng điều đó **không tự chứng minh đã giải repeated Bayesian equilibrium**.

Nếu bản đầu lặp một myopic stage decision rule, gọi đúng như vậy. Tên method ban đầu đề xuất:

> Local aggregation defense based on Bayesian best responses under a specified model-poisoning behavior model.

Lộ trình:

1. Hoàn thiện core DFL simulator và metrics.
2. Hoàn thiện Bayesian type/prior/likelihood/payoff formulation.
3. Derive, implement và validate local Bayesian best response.
4. Khi đủ cơ sở, formalize repeated Bayesian policies/histories, hoặc solve restricted episode-level Bayesian policy game; chọn target best response/equilibrium và kiểm chứng đúng target đó.
5. **Sau tuyến Bayesian trên**, mới cân nhắc Stackelberg/Markov hoặc các add-ons khác.

History, recursive beliefs và adaptive attack scheduling là những extensions có thể cần để nghiên cứu repeated behavior sâu hơn; không tự thêm chúng trước khi bản đầu chạy đúng. Bayesian current-evidence inference vẫn phải có trong initial Bayesian method, khác với history-based belief update.

Stackelberg cần policy commitment, follower knowledge và best-response timeline rõ ràng. “Gửi message trước, chọn weights sau” không đủ.

Markov/stochastic game cần state/transition đủ và solution concept tương ứng. Joint network state không phải global model, và không được tự cung cấp state đó cho local nodes.

## 15. Module boundaries để bắt đầu implementation

Các tên dưới đây là logical interfaces, không bắt buộc là final file/class names.

| Module                 | Trách nhiệm                                           | Không được làm                                  |
| ---------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| Episode sampler/config | Graph, partitions, seeds, types, attack configuration | Truyền true types vào local observations        |
| Benign learner         | Own model, local train/validation, aggregation        | Global aggregation hoặc đọc evaluator test data |
| Attacker controller    | Tạo outgoing manipulated message của compromised node | Sửa benign messages hoặc thay scheduler         |
| Scheduler/transport    | Snapshot rounds, delivery theo graph                  | Chọn defense weights thay nodes                 |
| Observation builder    | Chỉ tính local features                               | Đọc identities/metrics toàn mạng để tạo feature |
| Defense policy         | Observation + frozen artifact → weights               | Đọc full simulator object                       |
| Offline calibrator     | Likelihood/payoff tables và coverage checks           | Dùng final test results để tune                 |
| Evaluator/logger       | Privileged metrics, labels, trajectories              | Rò privileged fields sang policy                |

Baseline và Bayesian policies dùng chung observation/aggregation interfaces. Chạy simulator trên một máy không biến nó thành centralized FL nếu các boundaries trên được giữ.

Pseudocode online:

```text
for each synchronous round:
    trained_models = local_train_from_snapshots()
    outgoing_messages = honest_messages(trained_models)
    if episode_has_attacker:
        outgoing_messages[m] = attacker.manipulate(own_snapshot, trained_models[m])
    received = transport.deliver(outgoing_messages, graph)

    for each benign receiver i:
        observation = build_local_observation(i, received[i])
        weights = defense.select_weights(observation, frozen_artifact)
        next_model[i] = local_aggregate(trained_models[i], received[i], weights)

    if episode_has_attacker:
        next_model[m] = trained_models[m]
    commit_next_models_simultaneously()
    evaluator.log_round()             # privileged, không làm policy input
```

Code phải xử lý clean episode không có \(m\). Runtime guard có thể validate weights trước aggregation.

## 16. Implementation milestones và tiêu chí hoàn thành

### M0 — Core simulator, chưa có Bayesian defense

- Local-only và plain neighbor averaging.
- Two-node deterministic debug; three-node chain cho propagation.
- Clean và fixed-poison episodes.
- Logs: loss/accuracy, actual attack action, messages summary và weights.
- Unit tests protocol/information boundaries chạy đúng.

### M1 — Bayesian calibration và local best-response policy

- Episode/type/attack-strategy samplers rõ ràng.
- Current evidence, prior, likelihood và conditional payoff estimates.
- Finite action enumeration, posterior normalization, argmax/tie-breaking.
- Counterfactual calibration không thay actual trajectories.
- Exact small artificial game tests, tách khỏi empirical payoff claims.
- Frozen-artifact online execution.

### M2 — Kiểm chứng phương pháp trước add-ons

- Paired seeds/partitions, repeated runs và uncertainty reporting.
- So với averaging, fixed filter, local-only reference.
- So với policy chọn theo một fixed attack, giữ features/action space/tuning resources tương đương khi có thể.
- Kiểm tra influence của prior/behavior model lên decisions, cùng với task-performance trade-offs.
- Evaluate unseen intensities/placements; non-IID khi nằm trong experiment plan.
- Không dùng final test để chọn bins, intensities, policies hoặc baselines.

### M3 — Chỉ sau M0–M2

- Formal repeated Bayesian strategy analysis và best-response/equilibrium evidence đúng claim.
- History/recursive beliefs/adaptive scheduling nếu research question cần.
- Sau đó mới thêm Stackelberg commitment hoặc Markov state/transition formulation.
- Nhiều compromised nodes, independent attackers, recipient-specific messages và topology actions là extensions riêng, không một gói bắt buộc.

## 17. Tests và ba nhóm bằng chứng phải tách riêng

### A. Simulator correctness

- Synchronous snapshots: không dùng next-round models trong current round.
- Sender chỉ gửi đến graph neighbors; broadcast messages nhất quán.
- \(\kappa=0\) đúng bằng honest trained model; manipulation formula đúng trên small tensors.
- Attacker không mutate original benign/own snapshots qua tensor aliasing.
- Weight constraints, all-reject/self-only và missing-message handling.
- Deterministic seeds cho small reproducibility tests.
- Privileged evaluator fields không có trong policy interface.

### B. Derivation và solution quality

- Posterior đúng Bayes trên hand-computable tables, tổng bằng 1.
- Type joint support đúng constraint tối đa một compromised node.
- Selected action đạt maximum expected payoff trong candidate set, trừ documented tie tolerance/fallback.
- Same rule có thể tạo actions khác nhau khi observations thay đổi.
- Label rõ empirical/calibrated approximation, reference-behavior dependence và sparse-cell uncertainty.
- Nếu claim equilibrium sau này, có unilateral-deviation checks đúng utility/information/timing; restricted-space evidence không phải unrestricted theorem.

### C. Defense usefulness

- Clean utility không bị bỏ qua.
- Attack impact và multi-hop propagation được đo riêng.
- Non-IID false alarms/benign rejections được phân tích đúng nghĩa.
- Chi phí, performance distribution và sensitivity được báo cáo.
- Bayesian/game-derived policy phải được so sánh công bằng với simple rules.

Simulator chạy đúng không chứng minh defense tốt. Defense tốt không tự chứng minh equilibrium. Policy là best response trong model không bảo đảm tốt ngoài assumptions của model.

## 18. Các configuration cần chốt trước experiments

Không điền các giá trị chưa đo vào report như kết quả đã được suy ra. Lưu các lựa chọn sau trong reproducible config:

- Dataset, model architecture, preprocessing và local train/validation/design/test splits.
- IID/non-IID partition scheme, graph và placement distribution.
- Number of rounds, local steps, learning rate, optimizer và seeds.
- \(p_0\), attack intensity set, calibration strategy distribution \(\mu_A\).
- Reference weights, downweight multiplier \(\omega\), candidate ordering và tie tolerance.
- Feature normalization/bins, context bins, smoothing và unseen-cell fallback.
- Utility/cost definitions và coefficients nếu dùng.
- Calibration/selection/final-evaluation separation và number of repeated runs.

## 19. Giới hạn claim và câu mô tả khuyến nghị

Claim đầu tiên nên là:

> We formulate local model-poisoning defense in DFL as a Bayesian interaction with hidden neighbor types. Under a specified attacker-behavior model, we derive an approximate myopic best-response aggregation policy over a finite local action space. Each benign node executes the policy using only local observations and a frozen offline calibration artifact.

Chưa claim repeated Bayesian equilibrium, optimal long-horizon defense, robustness với mọi model poisoning, adaptive robustness hoặc fully local online learning.

Thông tin privilege dùng để calibration phải được công bố là **centralized offline design with decentralized execution**.

Tài liệu liên quan:

- `strategic_defense_research_direction.md`: mục tiêu defense algorithm và evaluation storyline; scope đầu được thu gọn ở tài liệu này.
- `dfl_game_modeling.md`: network dynamics, local information, contamination và các mở rộng state/Markov.
- `game_derived_policy_validation.md`: derivation, solution quality và empirical usefulness là ba claims riêng.
- `game_theory_research_note.md`: terminology và related-work context; không thay thế reading/verification trước khi reproduce papers.

**Tóm tắt:** Bayesian là information structure ngay từ đầu; best response là solution target đầu tiên. Xây simulator nhỏ và đúng trước, derive một local policy có thể kiểm chứng, rồi mới formalize repeated/equilibrium và thêm Stackelberg/Markov khi cần.
