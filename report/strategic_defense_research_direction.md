# Hướng nghiên cứu: strategic defense cho poisoning trong DFL

## 1. Đề tài và đóng góp chính

**Chống tấn công đầu độc mô hình trong Học Liên Kết Phi tập trung dựa trên mô hình trò chơi chiến lược.**

Đóng góp hướng tới là một **thuật toán phòng thủ cục bộ cho DFL**, được thiết kế từ tương tác chiến lược giữa benign nodes và attacker. Simulator chỉ là hạ tầng thực nghiệm, không phải sản phẩm nghiên cứu chính.

Pipeline nghiên cứu:

```text
Threat model DFL
    → Mô hình tương tác chiến lược
    → Local defense policy
    → Local aggregation
    → Đánh giá robustness, clean utility và chi phí
```

Câu hỏi trung tâm:

> Một benign node nên xử lý các neighbor updates thế nào khi attacker có thể thích nghi với defense, trong khi node chỉ có local information?

Ghi chú này điều chỉnh trọng tâm của kế hoạch implementation trước: mô hình hóa và mô phỏng là phương tiện để xây dựng defense, không phải điểm kết thúc.

## 2. Phân biệt game simulator và game-based defense

| Game simulator | Game-based defense — trọng tâm đề tài |
|---|---|
| Mô phỏng các bên chọn action | Thiết kế cách benign node chọn defense action |
| Quan sát trust/model/topology thay đổi | Dùng phân tích chiến lược để chọn aggregation weights |
| Tạo payoff matrix và khảo sát hành vi | Chuyển mô hình payoff và attacker responses thành thuật toán |
| Sản phẩm là môi trường mô phỏng | Sản phẩm là defense algorithm và bằng chứng thực nghiệm |

Payoff matrix, equilibrium analysis và simulator có thể hỗ trợ nghiên cứu. Tuy nhiên, nếu không dẫn đến một cơ chế phòng thủ triển khai được, chúng chưa đáp ứng đầy đủ mục tiêu đề tài.

## 3. Players, thông tin và actions

### Players

- Mỗi benign node là learner, sender, local aggregator và defender của neighborhood riêng.
- Một attacker controller điều khiển một compromised node trong phạm vi ban đầu.
- Có thể mở rộng số compromised nodes sau; không mặc nhiên xem từng node bị kiểm soát là một attacker độc lập.

### Thông tin

- Benign node chỉ dùng local model, local data, received messages và interaction history.
- Không cung cấp true malicious identities, global ASR hoặc toàn bộ network state cho local policy.
- Trust score không tự động là xác suất malicious; Bayesian belief cần observation/likelihood model được định nghĩa hoặc ước lượng.
- Benign sender có thể gửi model đã chịu ảnh hưởng poisoning. Hidden malicious type và model contamination là hai khái niệm khác nhau.

### Attacker actions

Ban đầu chọn một untargeted model-poisoning family, với cường độ và thời điểm thay đổi. Các policies kiểm tra có thể gồm constant-intensity, warm-up, on–off và adaptive-intensity attack.

Attacker chỉ được sửa thông điệp của node mình kiểm soát. Feedback và knowledge của attacker phải được quy định, không mặc nhiên toàn cục.

### Defender actions

Biến điều khiển chính là **aggregation weights**: reject, downweight hoặc accept theo mức tham chiếu của protocol. Threshold và trust có thể là thành phần hỗ trợ chọn weights, không nhất thiết là các action độc lập.

Giữ/loại kết nối và communication-frequency control để ngoài phạm vi ban đầu. Nếu thêm về sau, phải đánh giá mất collaboration và graph fragmentation.

## 4. Defense algorithm cần làm gì?

Mỗi benign node thực hiện tại mỗi round:

1. Nhận neighbor updates.
2. Tính evidence cục bộ: similarity, norm, validation signals và history.
3. Cập nhật memory và, nếu sử dụng Bayesian formulation, beliefs theo observation model.
4. Đánh giá candidate defense actions, có tính đến attacker behavior/response model hoặc một tập attacker strategies khả dĩ.
5. Chọn aggregation weights theo objective và ràng buộc đã xác định.
6. Aggregate local model và cập nhật evidence cho round tiếp theo.

**Đóng góp chủ yếu nằm ở bước 4–5.** Chỉ kết hợp scores thành trust rồi áp dụng threshold cố định chưa đủ để khẳng định một strategic-game defense.

Game formulation phải tạo ra khác biệt có thể kiểm tra được trong quyết định phòng thủ. Nếu attacker-response component không ảnh hưởng policy, game có nguy cơ chỉ là lớp diễn giải cho một rule-based filter.

## 5. Utility và reward

Defender cần cân bằng local task performance, collaboration utility và defense costs. Không nên thiết kế utility khiến reject toàn bộ updates luôn thắng một cách tầm thường.

Một local feedback minh họa, khi có local validation data, là:

\[
\widehat r_i^t=L_i^{val}(w_i^t)-L_i^{val}(w_i^{t+1})
-\lambda_c C_i^{comm,t}-\lambda_p C_i^{comp,t}.
\]

Feedback này không mặc nhiên đo được backdoor damage, true false positives hoặc network-wide propagation. Các đại lượng đó có thể được evaluator báo cáo riêng. Tránh đếm damage hai lần nếu task loss đã phản ánh cùng một ảnh hưởng.

Attacker objective có thể gồm damage tại mục tiêu, propagation impact và attack effort. Chỉ dùng network-wide metrics làm online feedback nếu threat model thực sự cho attacker truy cập; nếu dùng privileged simulator feedback để training, phải công bố rõ.

Với mục tiêu dài hạn:

\[
J_k=\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_k^t\right].
\]

Viết discounted utility chưa đủ để chứng minh thuật toán tối ưu dài hạn. Solver phải thực sự tính đến future effects nếu muốn đưa ra claim đó.

## 6. Game family và lựa chọn solver

Hướng xuất phát:

> Repeated Bayesian network game để thiết kế local aggregation defense trước adaptive poisoning.

- **Network:** interaction bị ràng buộc bởi communication graph; poisoning có thể lan nhiều hop.
- **Repeated:** actions và evidence tích lũy qua training rounds.
- **Bayesian:** hidden types và bất định giữa malicious behavior với benign non-IID heterogeneity; cần likelihood/belief model cụ thể.
- **Markov/stochastic:** thêm khi cần biểu diễn local model/trust evolution và hậu quả dài hạn. Joint network state không đồng nghĩa global model hoặc global observation.
- **Stackelberg:** chỉ thêm khi có commitment và follower-response timeline rõ ràng.
- **Evolutionary:** chưa cần nếu không nghiên cứu imitation, mutation hoặc population-selection dynamics. Model contamination lan truyền không đồng nghĩa strategy evolution.

Chưa chốt solver. Hai hướng có thể cân nhắc, chọn **một** làm phương pháp chính:

### A. Bayesian best-response defense

Defender tối ưu expected utility theo beliefs và một mô hình attacker behavior/response. Cần xác định cách ước lượng likelihood, payoffs và responses bằng dữ liệu nào. Nếu chỉ tối ưu immediate utility, không tuyên bố tối ưu dài hạn.

### B. Robust strategic defense

Defender tối ưu trước phản ứng bất lợi trong một tập attacker strategies có giới hạn. Cần attack budget/action constraints để bài toán có ý nghĩa; kiểm tra sự bảo thủ và clean-utility trade-off.

Finite policy games có thể dùng để kiểm tra ban đầu. Best response trong một policy set hữu hạn không phải equilibrium của toàn bộ DFL. Không mặc nhiên khẳng định convergence hoặc equilibrium khi chưa có phân tích/solver hỗ trợ.

## 7. Phạm vi implementation

### Công cụ

- Python: implementation và experiments.
- PyTorch: local training, update manipulation và evaluation.
- NetworkX: graph, routing neighborhoods và topology analysis.
- NumPy/pandas/Matplotlib: kết quả và plots.
- pytest: kiểm tra protocol, aggregation và information boundaries.
- Configuration, seeds và trajectories lưu bằng các định dạng đơn giản, tái lập được.

Một simulator synchronous nhỏ là đủ cho giai đoạn đầu. Chưa cần đưa Flower/FedML, distributed deployment hoặc deep RL vào critical path. Framework có thể cân nhắc sau nếu giải quyết được nhu cầu cụ thể.

Chạy toàn bộ simulator trên một máy không biến thuật toán thành centralized FL, miễn scheduler không thực hiện global aggregation thay nodes và không làm rò rỉ global information vào local policies.

### Phạm vi thí nghiệm

- **Hai node:** chỉ dùng debug/minimal illustration. Có cả clean và attacked episodes nếu muốn kiểm tra hidden-type uncertainty.
- **Ba node:** kiểm tra propagation từ malicious node qua benign intermediary đến benign node không trực tiếp kết nối attacker.
- **Mạng nhỏ, ví dụ 10–20 nodes:** thiết lập nghiên cứu ban đầu; một attacker kiểm soát một node, nhiều benign local defenders, fixed graph, synchronous rounds, IID và non-IID.

Chọn một untargeted attack family trước; thêm targeted/backdoor sau nếu đủ thời gian. Dùng cùng partitions, initializations và nhiều seeds cho các phương pháp so sánh.

## 8. Evaluation phải chứng minh strategic component có ích

### Baselines

- Local-only learning: đối chứng khi không collaboration, không gọi là DFL defense đầy đủ.
- Plain neighbor averaging.
- Fixed similarity filter.
- Trust-based weighting với rule cố định.
- BALANCE ở thiết lập mạng phù hợp; cân nhắc DFL-Dual nếu đủ nguồn lực tái hiện đúng.

### Experiments và ablations

| Thí nghiệm | Câu hỏi |
|---|---|
| Clean runs | Defense có tự làm giảm utility không? |
| Static poisoning | Robustness cơ bản ra sao? |
| On–off/adaptive intensity | Có chống được attacker thích nghi không? |
| Non-IID | Có nhầm heterogeneity với malicious behavior không? |
| Attacker placement/topology | Vị trí và đường lan truyền ảnh hưởng ra sao? |
| Bỏ attacker-response component | Strategic reasoning có đóng góp thật không? |
| Bỏ history/belief component | Repeated/Bayesian component có ích không? |

Để cô lập tác động của strategic reasoning, ablation nên giữ cùng evidence/features và aggregation interface khi có thể. Điều chỉnh hyperparameters công bằng trên validation setup; không dùng test results để chọn policy.

Metrics gồm local-model performance trên benign nodes, mean/worst-node performance, attack impact, propagation và computation/communication costs. Nếu đánh giá backdoor, thêm ASR. Evaluator có thể biết ground truth; local policy không mặc nhiên được biết.

## 9. Deliverables và giới hạn claim

1. Threat model và game formulation rõ ràng.
2. Local defense algorithm/pseudocode triển khai được.
3. Implementation tách node observations khỏi privileged evaluator state.
4. Baseline comparisons và adaptive-attack evaluation.
5. Ablations kiểm tra đóng góp của strategic component.
6. Phân tích trade-offs, information assumptions và limitations.

Không tuyên bố equilibrium, convergence guarantee, hoặc robustness với mọi attacker nếu chưa có kết quả tương ứng. Nếu defense không hơn phiên bản rule-based khi giữ cùng features, cần xem lại lợi ích của game formulation thay vì chỉ đổi tên thuật toán.

## 10. Giải thích bằng ví dụ đơn giản

Hãy tưởng tượng các bạn nhỏ cùng làm bài. Mỗi bạn có vở riêng và trao đổi gợi ý với bạn ngồi gần. Có một bạn cố tình đưa gợi ý sai, nhưng đôi khi lại đưa gợi ý đúng để được tin tưởng.

- Nhận mọi gợi ý: dễ bị lừa.
- Không nhận gợi ý nào: an toàn hơn nhưng mất cơ hội học từ các bạn tốt.
- Luôn dùng một luật cố định: bạn xấu có thể học cách né luật.

Đề tài muốn tạo một cách chọn gợi ý thông minh hơn: nhìn bài của mình, nhớ cách bạn kia từng hành xử, cân nhắc người đó có thể đổi cách lừa, rồi quyết định dùng nhiều, dùng ít hoặc bỏ qua gợi ý.

**Simulator là lớp học thử. Game model là cách suy nghĩ về hai bên. Defense algorithm là cách chọn gợi ý mà mỗi bạn thực sự dùng.**

Kết quả nghiên cứu cần cho thấy cách chọn mới giúp các bạn học tốt hơn khi có người cố lừa, chứ không chỉ dựng được một lớp học giả.
