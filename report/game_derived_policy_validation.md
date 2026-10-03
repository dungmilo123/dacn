# Defense policy suy ra từ game theory: biểu diễn và kiểm chứng

> Ghi chú bổ sung cho `strategic_defense_research_direction.md` và `dfl_game_modeling.md`. Mục tiêu là làm rõ mối quan hệ giữa game formulation, defense policy và bằng chứng cần trình bày trong report.

## 1. Sản phẩm cuối cùng của đề tài

Đề tài hướng tới một **phương pháp phòng thủ DFL**, cùng dạng đóng góp thuật toán như các phương pháp local robust aggregation: có giả định, công thức/pseudocode, implementation và evaluation. Simulator là hạ tầng để thiết kế và kiểm tra phương pháp, không phải đóng góp chính.

Điểm cần làm rõ của hướng game-based defense là cách policy được suy ra từ utilities, information structure và strategic behavior assumptions. Không bắt buộc đạt cùng theoretical guarantees với các paper dùng làm baseline; claims phải tương ứng với kết quả thực sự có.

## 2. Suy ra từ game không đồng nghĩa chạy game tại mỗi round

Defense policy **không bắt buộc giữ một game model hoặc chạy game solver lúc vận hành**.

Một pipeline hợp lệ là:

```text
Định nghĩa game
    → Giải bài toán chiến lược offline
    → Thu được defense policy
    → Mỗi node thực thi policy bằng local information
```

Policy cuối có thể là công thức, threshold, bảng tra, decision rule hoặc neural network. Một threshold đơn giản vẫn có thể là kết quả game-theoretic hợp lệ nếu có chuỗi suy dẫn rõ ràng.

Điều cần chứng minh là:

> Policy thực sự được suy ra từ game đã định nghĩa, không phải một policy có sẵn rồi được gắn thêm cách diễn giải game-theoretic.

Điều này làm rõ phát biểu trong ghi chú trước rằng defense phải “sử dụng tương tác chiến lược”: sự sử dụng có thể nằm ở **giai đoạn thiết kế/giải offline**, không nhất thiết nằm trong một module online riêng biệt.

Offline derivation không tự giải quyết mọi information constraint. Report vẫn phải nói rõ dữ liệu dùng để calibration/training và những gì từng node được truy cập khi execution.

## 3. Biểu diễn policy trong report

Ở node \(i\), round \(t\):

\[
\boldsymbol{\alpha}_i^t=\pi_i(x_i^t).
\]

- \(x_i^t\): local information hoặc summary của local history, gồm evidence, validation signals và memory/beliefs nếu sử dụng.
- \(\pi_i\): defense policy.
- \(\boldsymbol{\alpha}_i^t\): aggregation weights cho local model và received neighbor updates.

Với tập sender khả dụng \(R_i^t\):

\[
\alpha_{ij}^t\geq0,\qquad
\sum_{j\in R_i^t\cup\{i\}}\alpha_{ij}^t=1.
\]

Report cần nêu rõ input, output, feasible action space, cách cập nhật memory và quy tắc xử lý missing messages. Policy là quy tắc chọn action trong các tình huống khác nhau, không chỉ là một vector weights của một round cụ thể.

Nếu dùng biểu diễn optimization:

\[
\pi_i(x)=\arg\max_{\boldsymbol{\alpha}\in\mathcal A_i}
\widehat Q_i(x,\boldsymbol{\alpha}),
\]

thì phải định nghĩa cụ thể \(\mathcal A_i\), \(\widehat Q_i\), estimation/training procedure và tie-breaking nếu cần. Công thức argmax đơn thuần chưa đủ để tái hiện phương pháp, cũng chưa tự chứng minh strategic reasoning.

## 4. Chuỗi suy dẫn cần trình bày

### 4.1. Định nghĩa game

- Players và private information.
- Actions và strategy/policy spaces.
- Utilities, horizon và discounting nếu có.
- Timing, observations và commitment nếu có.
- Constraints và knowledge của attacker/defender.
- Solution concept được chọn.

### 4.2. Điều kiện nghiệm

Ví dụ, với Nash-type equilibrium, các policies phải là best responses tương ứng:

\[
\pi_i^*\in\arg\max_{\pi_i\in\Pi_i}
J_i(\pi_i,\pi_{-i}^*).
\]

Biểu thức này chỉ minh họa cho Nash-type solution, không áp dụng mặc định cho mọi game. Hidden-type formulations cần expectations và information constraints phù hợp. Sequential games có thể cần sequential rationality và belief consistency; không thay thế những điều kiện đó bằng một kiểm tra ex-ante đơn giản rồi claim một sequential equilibrium.

Nếu chọn Stackelberg, phải biểu diễn commitment và follower best-response đúng timeline. Nếu chọn robust optimization, gọi đúng là robust solution; không mặc nhiên gọi đó là Nash equilibrium.

### 4.3. Thuật toán tìm policy

Nêu rõ phương pháp:

- Analytical derivation.
- Numerical optimization.
- Search/solution trong finite policy set.
- Learned approximation.

Phân biệt policy là nghiệm chính xác, nghiệm xấp xỉ hay heuristic được game formulation hướng dẫn. Không dùng cùng một claim cho ba mức này.

Chuỗi cần theo dõi được:

\[
\text{utilities + information + response assumptions}
\longrightarrow\text{solution conditions}
\longrightarrow\text{solver/derivation}
\longrightarrow\pi^*.
\]

## 5. Ba loại bằng chứng khác nhau

### A. Quan hệ suy dẫn — policy có nguồn gốc game-theoretic không?

Trình bày công thức và procedure nối game formulation với policy cuối. Một theorem lớn không phải điều kiện bắt buộc cho mọi phạm vi nghiên cứu, nhưng phương pháp phải đủ cụ thể để người khác tái hiện và kiểm tra assumptions.

Không đủ nếu chỉ mô tả một filter có sẵn bằng ngôn ngữ payoff mà không chỉ ra cách game ảnh hưởng đến việc thiết kế/tìm policy.

### B. Chất lượng nghiệm — policy giải game đến mức nào?

Nếu claim approximate Nash equilibrium, một đại lượng có thể kiểm tra là unilateral deviation gain:

\[
\epsilon_i=
\max_{\pi_i'\in\Pi_i}J_i(\pi_i',\pi_{-i}^*)
-J_i(\pi_i^*,\pi_{-i}^*).
\]

Giá trị nhỏ biểu thị khó cải thiện utility bằng unilateral deviation trong strategy space đang xét. Utilities và deviations phải tuân thủ cùng information/timing constraints với game ban đầu.

Nếu chỉ thử một số candidate policies, kết quả là empirical/restricted-space evidence, không phải chứng minh equilibrium trên mọi feasible policy. Monte Carlo estimates cũng có sampling uncertainty; nên dùng repeated runs và báo cáo độ bất định phù hợp.

Không bắt buộc dùng equilibrium claim. Nếu phương pháp chỉ claim best-response defense hoặc robust optimization, kiểm chứng theo objective và solution concept tương ứng thay vì thêm một equilibrium metric không phù hợp.

### C. Hiệu quả phòng thủ — policy có hữu ích trong DFL không?

Đánh giá task performance, attack impact, false positives khi có ground truth, defense costs và adaptive robustness so với baselines. Tách calibration/policy selection khỏi final evaluation để tránh chỉ tối ưu trên các tình huống đã dùng thiết kế phương pháp.

**Adaptive robustness tốt không tự chứng minh game-theoretic derivation. Ngược lại, một nghiệm hợp lệ của game không tự bảo đảm hiệu quả ngoài assumptions của game.**

Ba câu hỏi A, B, C là ba claims riêng biệt và cần bằng chứng riêng.

## 6. Ablation khi game được giải offline

Ablation “bỏ strategic component” không nhất thiết là tháo một module khỏi policy khi runtime. Có thể phải **suy ra hoặc huấn luyện lại policy dưới các assumptions/objectives khác**.

| Policy chính | Đối chứng có thể dùng |
|---|---|
| Tính đến attacker thay đổi strategy | Giả định attacker behavior cố định |
| Tính đến future consequences | Chỉ tối ưu immediate utility |
| Sử dụng hidden-type beliefs | Không sử dụng belief component |

Chỉ thực hiện ablations tương ứng với components thực sự có trong phương pháp. Giữ features, feasible actions, data và tuning resources tương đương khi có thể; giải thích nếu không thể cô lập hoàn toàn.

Mục tiêu là kiểm tra lợi ích của strategic assumptions/optimization, không phải chứng minh policy online phải chứa một game solver. Policy được compile thành threshold hoặc lookup table vẫn có thể được so sánh theo cách này.

## 7. Cách trình bày thuật toán triển khai

Report nên có:

1. Policy equation và input/output definitions.
2. Offline derivation/training algorithm, nếu có.
3. Online execution pseudocode cho benign node.
4. Complexity và information requirements.
5. Một local trajectory minh họa evidence, action và outcomes.

Phân biệt minh họa giả định với kết quả thực nghiệm. Không dùng ví dụ weights/thresholds chưa được tính để trình bày như policy đã suy ra hoặc đã được chứng minh hiệu quả.

Nếu policy là learned model, nêu architecture, training objective, datasets/episodes, selection procedure và inference process. Không bắt buộc rút policy thành một luật ngắn nếu cách biểu diễn đó không phản ánh đúng phương pháp.

## 8. Phát biểu có thể dùng trong report

> We formulate the interaction as a strategic game and derive a local defense policy from its solution. During execution, each node applies the resulting policy using only local observations; online equilibrium computation is not required.

Chỉ dùng “from its solution” khi đã xác định solution concept và có kết quả giải tương ứng. Nếu mới là heuristic, dùng cách diễn đạt thận trọng hơn như “guided by the strategic formulation”.

## 9. Tóm tắt

**Không cần chứng minh defense policy luôn dùng game model lúc chạy.** Cần làm rõ:

- Policy có nguồn gốc từ game formulation như thế nào.
- Policy giải bài toán chiến lược đến mức nào, trong assumptions và strategy space nào.
- Policy có hiệu quả phòng thủ thực nghiệm ra sao.

Sản phẩm cuối là **policy được định nghĩa rõ + thuật toán triển khai + bằng chứng tương ứng với claims**, không phải simulator hoặc nhãn game-theoretic đơn thuần.
