# Game formulation cho poisoning trong DFL: player, action, strategy, utility, type, state

> Ghi chú thiết kế nghiên cứu, bổ sung cho `dfl_game_modeling.md`, `game_theory_research_note.md` và `game_derived_policy_validation.md`. Nội dung dưới đây là **đề xuất mô hình hóa** dựa trên các tài liệu đó và implementation thực tế trong `src/dfl_sim/`; chưa phải thuật toán hay equilibrium đã được chứng minh. Tên class/field trong code được nêu để đối chiếu, không thay thế định nghĩa hình thức.

## 0. Bối cảnh và phạm vi

Mô hình đề xuất là **repeated Bayesian interaction on a communication graph**:

- Không có central server. Defender là các **benign receiving nodes**, không phải một aggregator trung tâm.
- Mỗi node ra quyết định bằng local information và lịch sử tương tác của chính mình.
- Joint state mô tả cách toàn mạng cùng tiến triển; nó **không** được giả định là quan sát được bởi từng node.
- Một Markov abstraction có thể được dùng để huấn luyện policy, nhưng giới hạn thông tin và mức xấp xỉ phải được nêu rõ.

Formulation tối thiểu phải định nghĩa: **players, state, private information/type, actions, observation, transition, utility/payoff, solution concept**. Thiếu các định nghĩa này thì một score/threshold/weighted-average chỉ là **defense rule**, chưa phải **strategic game model**.

---

## 1. Player

Player phải là **decision-maker thật**, có action và objective. Model, aggregate, score hay trust value là **object** trong game, không phải player.

| Player | Vai trò trong project | Ghi chú |
|---|---|---|
| **Benign receiving node** \(i\) | Learner, sender, local aggregator và defender | Có thể chọn aggregation weights cho neighbor; mở rộng sang reject/isolation nếu protocol cho phép. Nhiều node dùng chung policy architecture nhưng **thực thi độc lập** trên local observations. Shared policy **không** đồng nghĩa centralized defender. |
| **Attacker** | **Một** decision-maker điều khiển tập node bị compromise \(M\subseteq V\) | Không nhất thiết coi từng compromised node là player độc lập. Trong code: `AdaptiveAttackController` là một controller, `compromised_node` là "tay chân". |

Không mặc nhiên cho attacker biết toàn bộ graph hoặc state. **Attacker knowledge là một phần riêng của threat model**, phải khai báo.

---

## 2. Action

**Action** là một lựa chọn khả thi tại một decision point. **Action space** là tập hợp các lựa chọn đó.

### 2.1. Attacker action

Trong implementation hiện tại, attack là một transform one-shot của outgoing model:

\[
u = \tilde w - \kappa\,(\tilde w - w)
\]

với \(\kappa\) chọn từ bảng D016 (`ATTACK_KAPPAS`). Attack family ban đầu gồm **honest / silent / poison** với một tập intensity hữu hạn. `kappa == 0.0` là no-op tensor-identical với honest.

Về lý thuyết có thể mở rộng: attack timing, intensity, target label, data poisoning, model manipulation, recipient-specific messages. Nếu protocol broadcast một thông điệp giống nhau cho mọi neighbor, action space phải tuân theo ràng buộc đó.

### 2.2. Defender action

Defender chọn aggregation weights:

\[
d_i^t=\{\alpha_{ij}^t\}_{j\in R_i^t\cup\{i\}},\qquad
\alpha_{ij}^t\ge 0,\qquad
\sum_{j\in R_i^t\cup\{i\}}\alpha_{ij}^t=1
\]

Code biểu diễn dưới dạng `WeightProfile` với `multipliers`:

- self multiplier **cố định bằng 1**;
- neighbor multiplier \(\in\{0,\ \omega,\ 1\}\) = **reject / downweight / accept** (D017);
- reject toàn bộ neighbor → exact self-only weights \((1.0, 0.0, \dots)\).

`build_weight_profile` fold multiplier vector thành normalized weights mà **không** enumerate \(3^{\text{degree}}\) joint action space (D017); enumeration có pruning qua `degree_cap` (`enumerate_candidate_actions`).

---

## 3. Strategy / Policy

**Strategy** chỉ định action cho **mọi** tình huống có thể gặp:

- **pure**: chọn một action deterministically;
- **mixed**: chọn action theo phân phối xác suất;
- **policy**: ánh xạ observation/state → action (dùng cho repeated/Markov game).

Trong code, seam chính là:

```python
class AggregationPolicy(Protocol):
    def select_weights(self, observation: LocalObservation) -> WeightProfile: ...
```

Các policy hiện có: `LocalOnlyPolicy`, `PlainAveragePolicy`, `FixedSimilarityFilterPolicy`, `ValidationReferencePolicy`, `TrustWeightedPolicy`, `BalancePolicy`, `BayesianPolicy`.

`BayesianPolicy` là **exact approximate-myopic Bayesian best response**: tính posterior trên các feasible joint types theo artifact, tính posterior-weighted expected payoff của mọi candidate action (D022), và chọn argmax đầu tiên theo tie tolerance.

### 3.1. Information boundary — policy không đọc global state

Policy chỉ đọc `LocalObservation` — seam bị khóa bằng **exact-type guard** (`type(observation) is LocalObservation`, không dùng `isinstance`). Điều này ngăn một subclass reintroduce mutable `__dict__` và tuồn ground-truth/evaluator sentinel qua seam.

Mỗi node dùng:

\[
d_i^t\sim\pi_i^D(\cdot\mid h_i^t),\qquad
x_i^t=\phi_i(h_i^t),\qquad
d_i^t\sim\pi_i^D(\cdot\mid x_i^t)
\]

trong đó \(h_i^t\) là observations và actions đã biết của node \(i\); \(x_i^t\) có thể chứa trust vector, recent scores, local loss, available neighbors và recurrent memory.

`LocalObservation` chứa scalar context và per-neighbor feature triple:

\[
\text{disagreement}=1-\text{clamp}(\cos(\delta_{ii},\delta_{ij})),\quad
\text{relative\_norm}=\frac{\lVert\delta_{ij}\rVert}{\max(\lVert\delta_{ii}\rVert,\epsilon)},\quad
\text{loss\_delta}=L_i(u_{j\to i})-L_i(\tilde w_i)
\]

**Cảnh báo**: toàn mạng Markov **không** bảo đảm local observation Markov. Node ngoài neighborhood có thể thay đổi model của neighbor, khiến phân phối tương lai khác nhau dù local features hiện tại giống nhau. Vì vậy \(x_i^t\) là **policy input**, không mặc nhiên là exact Markov state.

---

## 4. Utility / Payoff

Utility chuyển outcome thành thang ưu tiên số học cho mỗi player. Trong security thường là **multi-objective**.

### 4.1. Defender utility

Với local validation set:

\[
r_i^t=L_i^{val}(w_i^t)-L_i^{val}(w_i^{t+1})-\lambda_c C_i^{comm,t}-\lambda_p C_i^{comp,t}
\]

\[
J_i=\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_i^t\right]
\]

Reward này phản ánh local improvement và chi phí, nhưng **không** đảm bảo đo được backdoor damage, global convergence hay đóng góp riêng của từng neighbor.

### 4.2. Attacker utility

Attacker tối ưu damage/ASR trên mục tiêu và dữ liệu nó truy cập được, trừ attack effort. Giá trị của việc duy trì trust xuất hiện qua future attack opportunities trong discounted utility — không nhất thiết cần thêm trust bonus riêng.

### 4.3. Payoff dạng bảng (Bayesian artifact)

Trong artifact rời rạc:

\[
q(x_j,\ c,\ d_j,\ \theta_j)
\]

với \(x_j\) = evidence bins (disagreement, relative_norm, loss_delta), \(c\) = context (degree bin, progress), \(d_j\) = multiplier action, \(\theta_j\) = type. Joint payoff **tách được**:

\[
Q(x,c,d,\theta)=\sum_j q(x_j,c,d_j,\theta_j)
\]

### 4.4. Ba khái niệm phải phân biệt

- **Objective dùng để mô hình hóa preferences**;
- **Feedback node thực sự nhận được**;
- **Metric evaluator báo cáo**.

True FPR, contamination, ASR là **latent/evaluation quantities**. Không dùng chúng như feedback quan sát trực tiếp nếu threat model không cho phép. Nếu dùng privileged simulator information để huấn luyện, phải gọi đúng là **centralized training with decentralized execution**, không phải fully local online learning.

> Utility là **giả định mô hình hóa**, không tự suy ra từ hệ thống. Phải khai báo tường minh khi so sánh paper hoặc tái lập kết quả.

---

## 5. Type (private information)

**Type** là đặc tính ẩn mà player khác không quan sát được. Trong code:

\[
\theta_j\in\{0,1\}=\{\text{benign},\text{malicious}\}
\]

Prior nằm trên **feasible joint states** \(\theta\in\{0,1\}^d\), với ràng buộc **at most one malicious neighbor** theo artifact hiện tại.

Node duy trì belief và cập nhật Bayes:

\[
b_{ij}^t=\Pr(\theta_j=\text{malicious}\mid h_i^t),\qquad
\text{posterior}(\theta)\propto\text{prior}(\theta)\cdot\prod_j L(x_j\mid\theta_j,c)
\]

### 5.1. Ba khái niệm không được đồng nhất

| Khái niệm | Ý nghĩa |
|---|---|
| **Type** | Có *chủ đích* tấn công hay không |
| **Action** | Đang gửi trung thực / im lặng / poisoning |
| **Model contamination** | Model hiện tại chịu ảnh hưởng poisoning đến mức nào |

Một node benign có thể nhận poisoned update rồi forward model đã nhiễm độc — **vẫn benign về ý định**.

### 5.2. Cảnh báo về belief

- Trust score **chưa calibrated** không nên gọi trực tiếp là xác suất malicious.
- Belief về type neighbor **không** mặc nhiên là sufficient statistic cho toàn bộ hidden network state. Trong multi-agent setting, diễn biến còn phụ thuộc policies và private information của các node khác.
- **Không** khẳng định một vector type beliefs tự động biến local problem thành fully observed Markov game.

---

## 6. State

### 6.1. Ba khái niệm không nên đồng nhất

| Khái niệm | Ý nghĩa trong DFL | Node truy cập? |
|---|---|---|
| Local model \(w_i^t\) | Model hiện tại của node \(i\) | Có, với model của chính nó |
| Joint network state \(S_t\) | Trạng thái chung của toàn mạng | Thường **không** |
| Global model \(\bar w_t\) | Model chung do protocol duy trì hoặc evaluator tổng hợp | **Không** mặc nhiên tồn tại/được cấp |

Một Markov game có thể có \(S_t\) mà **không** có \(\bar w_t\). Từ "global" trong global state nói về **phạm vi mô tả**, không nói về quyền truy cập hay kiến trúc điều phối.

### 6.2. Joint network state (template)

\[
S_t=\left(G_t,\ \{w_i^t,z_i^t\}_{i\in V},\ \{\tau_{ij}^t,m_{ij}^t\}_{(j,i)\in E_t},\ \xi_t^A\right)
\]

- \(w_i^t\): local model parameters;
- \(z_i^t\): optimizer/local training state (nếu cần cho transition);
- \(\tau_{ij}^t\): trust receiver \(i\) dành cho sender \(j\) — **có hướng**, không nhất thiết đối xứng;
- \(m_{ij}^t\): memory cho scoring và trust update;
- \(\xi_t^A\): attacker budget, node-control state hoặc adaptive memory.

### 6.3. Markov property

\[
\Pr(S_{t+1}\mid S_{0:t},\mathbf a_{0:t})=\Pr(S_{t+1}\mid S_t,\mathbf a_t)
\]

với environment/type parameters cố định được hiểu là đã điều kiện hóa.

State trên chỉ là **template**: cần bổ sung/lược bỏ biến theo protocol thực tế. Không cần lưu toàn bộ lịch sử nếu thuật toán chỉ dùng một bộ nhớ đủ đã xác định.

### 6.4. Transition được tạo bởi các bước cục bộ

Ví dụ synchronous rounds + model exchange. Node \(j\) local train:

\[
\tilde w_j^t=\operatorname{LocalTrain}_j(w_j^t,D_j,z_j^t)
\]

Thông điệp gửi đến \(i\):

\[
u_{j\to i}^t=
\begin{cases}
\tilde w_j^t,&\text{gửi trung thực},\\
\operatorname{Manipulate}_{j\to i}(\tilde w_j^t,a_j^t),&\text{model poisoning}.
\end{cases}
\]

Node \(i\) aggregate:

\[
w_i^{t+1}=\alpha_{ii}^t\tilde w_i^t+\sum_{j\in R_i^t}\alpha_{ij}^t u_{j\to i}^t
\]

Ghép các bước cục bộ:

\[
S_{t+1}\sim P(\cdot\mid S_t,\mathbf a_t^A,\mathbf d_t^D,\theta)
\]

Không có server thực hiện joint transition — nó là **mô tả chung** của các bước phân tán. Poisoning lan nhiều hop khi local model bị ảnh hưởng tiếp tục được gửi ở các round sau.

### 6.5. State abstraction

Accuracy, trust và similarity **chưa mặc nhiên** là exact Markov state: hai mạng có cùng summary metrics vẫn có thể có parameters khác nhau và chuyển sang trạng thái khác nhau sau cùng một action. Nếu dùng summary features, nên gọi là **state abstraction** hoặc **approximate Markov representation**.

---

## 7. Solution concept

Equilibrium là **dự đoán hình thức về hành vi ổn định** dưới một bộ players, utilities, information, action sets và timing cụ thể — **không** phải bảo đảm an toàn phổ quát.

### 7.1. Các solution concept và mức phù hợp

| Solution concept | Ý nghĩa | Phù hợp khi | Hạn chế trong bối cảnh DFL |
|---|---|---|---|
| **Nash equilibrium** | Mỗi strategy là best response với các strategy còn lại; không ai cải thiện utility bằng đổi strategy một mình | Game simultaneous-move, một-round abstraction | Không biểu diễn được trust accumulation, delayed attack, adaptation nhiều round |
| **Bayesian Nash equilibrium (BNE)** | Nash mở rộng cho hidden types; mỗi player chọn strategy theo type của mình và belief về type người khác | Defender không biết client benign hay malicious | Belief phải có likelihood/observation model phù hợp; không chỉ cộng trừ trust |
| **Stackelberg equilibrium** | Leader commit strategy trước; follower quan sát/suy đoán rồi best-respond | Có **policy commitment rõ ràng** trước round | **Không** tự động đúng chỉ vì defender phản ứng sau khi nhận update |
| **Minimax** | Chọn defense maximize outcome dưới worst plausible response của attacker | Game zero-sum/strictly competitive | Poisoning thật thường **không** zero-sum (clean performance, false positives, effort) |
| **Markov-game equilibrium** | Equilibrium policy profile trên state; action sinh ra state sau | Cả hai bên adapt lặp lại, reward tích lũy theo thời gian | Full multi-agent Markov game có thể quá lớn cho phạm vi luận văn |

### 7.2. Điều kiện nghiệm

Với Nash-type equilibrium:

\[
\pi_i^*\in\arg\max_{\pi_i\in\Pi_i}J_i(\pi_i,\pi_{-i}^*)
\]

Biểu thức chỉ minh họa cho Nash-type. Hidden-type formulation cần expectation và information constraints phù hợp; sequential game có thể cần **sequential rationality và belief consistency**, không thay thế bằng một ex-ante check rồi claim sequential equilibrium.

### 7.3. Vì sao bắt đầu từ repeated Bayesian rồi mới thêm Markov

1. **Repeated Bayesian network game** là hướng khởi đầu: biểu diễn incomplete information (hidden type) và trust accumulation qua nhiều round mà không cần full state-space machinery.
2. Bổ sung **state transitions** để nghiên cứu hậu quả dài hạn → **partially observable networked stochastic game with private node types**.
3. **Chỉ thêm Stackelberg** khi có giả định policy commitment rõ ràng.

Timeline hiện tại:

\[
\text{gửi update}\rightarrow\text{quan sát}\rightarrow\text{chọn defense weight}
\]

trước hết là **sequential interaction with incomplete information**. Với lịch sử, beliefs và sequential rationality được định nghĩa đầy đủ, có thể xem xét sequential Bayesian solution concepts; **chưa** nên khẳng định một equilibrium cụ thể khi chưa có solver/analysis.

### 7.4. Khi nào Stackelberg hợp lệ

Stackelberg phù hợp nếu có policy commitment rõ ràng, ví dụ:

\[
\text{defense policy được cam kết trước}\rightarrow\text{attacker quan sát/suy đoán}\rightarrow\text{attacker best-responds}
\]

Trong DFL phải giải thích: **ai** cam kết, cam kết **điều gì**, follower **biết gì**, và các benign nodes có phối hợp hay không. Không mặc nhiên biến toàn bộ benign nodes thành một central leader. Nếu defender cũng liên tục đổi policy sau khi quan sát attacker, phải mô tả **dynamic timing** thay vì dán nhãn Stackelberg cho thứ tự sự kiện đơn thuần.

### 7.5. Quan hệ giữa các local games

Tương tác receiver \(i\) và sender \(j\) có thể phân tích như một **local subgame**, nhưng các edge games **không độc lập**:

- weights của mọi neighbor cùng chịu ràng buộc chuẩn hóa;
- các update được aggregate cùng nhau, có tương tác về tác động;
- local model sau aggregation ảnh hưởng thông điệp gửi ở round sau;
- topology quyết định đường lan truyền và cơ hội tương tác tiếp theo.

Vì vậy mô hình hai-player cục bộ là **baseline/approximation**; không nên tuyên bố equilibrium của từng cạnh tự động là equilibrium toàn mạng.

### 7.6. Ba loại bằng chứng khác nhau

Xem chi tiết trong `game_derived_policy_validation.md`. Tóm tắt:

- **A. Quan hệ suy dẫn** — policy có nguồn gốc game-theoretic không (công thức + procedure nối formulation với policy cuối).
- **B. Chất lượng nghiệm** — policy giải game đến mức nào (ví dụ unilateral deviation gain \(\epsilon_i\) nếu claim approximate Nash).
- **C. Hiệu quả phòng thủ** — policy hữu ích trong DFL không (task performance, attack impact, false positives, cost, adaptive robustness).

Ba câu hỏi là ba claims riêng biệt, cần bằng chứng riêng. **Adaptive robustness tốt không tự chứng minh game-theoretic derivation; ngược lại, nghiệm hợp lệ của game không tự bảo đảm hiệu quả ngoài assumptions của game.**

---

## 8. Tổng hợp

| Element | Ý nghĩa | Ví dụ trong project |
|---|---|---|
| **Player** | Decision-maker thật, có action và objective | benign receiving node; attacker controller |
| **Action** | Một lựa chọn tại một decision point | attacker: honest/silent/poison + \(\kappa\); defender: accept/downweight/reject |
| **Strategy/Policy** | Ánh xạ tình huống → action | `select_weights(LocalObservation) -> WeightProfile` |
| **Utility/Payoff** | Lý do thích một action | \(r_i^t\) local improvement − cost; attacker damage − effort; bảng \(q(x_j,c,d_j,\theta_j)\) |
| **Type** | Thông tin ẩn | \(\theta_j\in\{0,1\}\); prior × likelihood → posterior |
| **State** | Thông tin mang qua round | \(S_t=(G_t,\{w,z\},\{\tau,m\},\xi^A)\); local input \(o_i^t\neq S_t\) |
| **Solution concept** | Nghĩa của "hành vi ổn định chiến lược" | BNE (khởi đầu) → Markov-game equilibrium (mở rộng); Stackelberg chỉ khi có commitment |

### Phạm vi triển khai đề xuất

1. Graph cố định và synchronous communication rounds.
2. Một attacker kiểm soát tập node cố định; phân biệt attacker controller với compromised nodes.
3. Benign nodes thực thi local defense policies; không truy cập global state hoặc malicious identities.
4. Attack actions ban đầu gồm honest/silent/poison với tập intensity hữu hạn; defender chọn reject/downweight/accept qua aggregation weights.
5. Ghi rõ self-weight, missing-message handling, local validation assumptions và việc có cho phép recipient-specific messages hay không.
6. Dùng local trust/history/features làm policy input; không khẳng định exact Markov sufficiency nếu chưa chứng minh.
7. Giữ attacker hidden types, model contamination và suspicious observations là các khái niệm khác nhau.
8. Đánh giá clean utility, attack impact, false positives, propagation nhiều hop và defense costs bằng evaluator tách biệt.
9. Bắt đầu từ repeated Bayesian network game; thêm Markov structure để nghiên cứu hậu quả dài hạn; chỉ thêm Stackelberg nếu commitment/best-response là research question thực sự.

### Câu mô tả mô hình có thể dùng trong luận văn

> We model poisoning in DFL as a repeated Bayesian interaction on a communication graph. Each benign node selects local aggregation actions using only its observations and interaction history, while an attacker controls the messages of a subset of nodes. The joint evolution of local models, directed trust and communication states defines network-level stochastic dynamics; this state is not assumed to be observable by individual nodes. A Markov abstraction may be used for tractable policy learning, with its information and approximation limits stated explicitly.

**Tóm lại:** local nodes không cần "lấy thông tin từ global model" để tham gia Markov game. Chúng tác động lên một hệ thống có joint state bằng actions dựa trên local information; joint state chỉ mô tả cách toàn mạng cùng tiến triển.
