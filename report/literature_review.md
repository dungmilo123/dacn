# Literature Review Draft

## Chống tấn công đầu độc mô hình trong Học Liên Kết Phi tập trung dựa trên mô hình trò chơi chiến lược

> Bản nháp 0.1. Mục tiêu của bản này là dựng được mạch lập luận, nhóm tài liệu, research gap và hướng mô hình hóa để tiếp tục mở rộng thành chương Literature Review chính thức.

## 1. Phạm vi và cách tổ chức literature review

Đề tài tập trung vào **Decentralized Federated Learning (DFL)**, trong đó các agent trao đổi trực tiếp model update với các neighbor trên một đồ thị giao tiếp thay vì gửi toàn bộ update về một server trung tâm. Việc loại bỏ server giúp giảm phụ thuộc vào một thực thể duy nhất, nhưng đồng thời khiến mỗi agent phải tự quyết định update nào đáng tin cậy và neighbor nào nên tiếp tục tương tác. Đây là điểm làm cho bài toán poisoning trong DFL khác đáng kể so với Federated Learning tập trung.

Literature review nên được tổ chức theo một mạch đi từ nền tảng đến khoảng trống nghiên cứu:

1. Federated Learning và sự chuyển dịch từ kiến trúc tập trung sang DFL.
2. Threat model và các dạng poisoning trong FL/DFL.
3. Tấn công poisoning đặc thù cho topology peer-to-peer.
4. Các defense hiện có trong DFL: local robust aggregation, trust/reputation, anomaly detection, topology-aware defense.
5. Các nghiên cứu sử dụng game theory để mô hình hóa tấn công-phòng thủ trong FL.
6. Research gap: thiếu cơ chế defense trong DFL coi cả attacker và defender là các tác nhân chiến lược, thích nghi theo thời gian và topology.

Mạch này quan trọng hơn việc đơn thuần liệt kê paper theo năm. Literature review cần trả lời được ba câu: **đã có gì, còn thiếu gì, và vì sao khoảng trống đó dẫn hợp lý tới đề tài hiện tại**.

---

## 2. Từ Federated Learning tập trung đến Decentralized Federated Learning

Federated Learning (FL) cho phép nhiều client huấn luyện mô hình chung mà không truyền raw data về một nơi tập trung. Trong kiến trúc phổ biến, một central server phân phối global model, nhận local update và thực hiện aggregation. Mô hình này tạo ra một điểm điều phối rõ ràng nhưng đồng thời tạo ra trust dependency, communication bottleneck và single point of failure.

Các nghiên cứu ban đầu về học liên kết peer-to-peer đã cho thấy có thể thay server bằng giao tiếp giữa các node trên graph. Lalitha và cộng sự đề xuất **Peer-to-peer Federated Learning on Graphs**, trong đó node kết hợp thông tin từ one-hop neighbors để học mô hình chung trên toàn mạng. BrainTorrent sau đó minh họa một môi trường FL không server trong bài toán medical imaging, nơi các trung tâm trao đổi trực tiếp model với nhau.

Hai công trình này cho thấy một nguyên tắc cốt lõi của DFL: trạng thái mô hình của mỗi node chịu ảnh hưởng chủ yếu từ **local neighborhood** và cấu trúc graph. Vì vậy, khác với FL tập trung nơi server nhìn thấy tập update tương đối toàn cục, một node DFL chỉ có quan sát cục bộ. Điều này làm cho security decision cũng buộc phải được thực hiện cục bộ.

Nguồn nền tảng:

- A. Lalitha, O. C. Kilinc, T. Javidi, F. Koushanfar, *Peer-to-peer Federated Learning on Graphs*, 2019. https://arxiv.org/abs/1901.11173
- A. Guha Roy et al., *BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated Learning*, 2019. https://arxiv.org/abs/1905.06731
- E. Hallaji et al., *Decentralized Federated Learning: A Survey on Security and Privacy*, 2024. https://arxiv.org/abs/2401.17319

---

## 3. Poisoning trong FL và điểm khác biệt của DFL

Poisoning attack nhắm vào quá trình huấn luyện bằng cách làm sai lệch dữ liệu hoặc update được đóng góp bởi client độc hại. Cần phân biệt rõ thuật ngữ trong đề tài:

- **Model poisoning**: attacker trực tiếp thao túng gradient, weight hoặc model update trước khi gửi đi, ví dụ sign flipping, scaling, gradient manipulation hoặc optimized malicious update.
- **Data poisoning**: attacker thay đổi local training data, ví dụ label flipping hoặc chèn poisoned samples.
- **Backdoor attack**: có thể được hiện thực bằng data poisoning hoặc model poisoning. Mục tiêu thường là giữ clean accuracy tương đối ổn nhưng làm mô hình phản ứng sai trên input có trigger.

Điểm này cần viết rõ trong luận văn vì danh sách thực nghiệm hiện tại có cả `gradient manipulation`, `label flipping` và `backdoor`. Nếu tiêu đề giữ cụm “model poisoning”, label flipping nên được mô tả là một **poisoning stress case bổ sung**, hoặc scope nên mở rộng thành “poisoning attacks” nếu muốn bao phủ cả data poisoning một cách chính thức.

Trong FL tập trung, server có thể áp dụng robust aggregation hoặc malicious-client detection trên tập update của nhiều client. Trong DFL, mỗi benign node chỉ nhận update từ neighbor của mình. Một attacker có thể gửi update khác nhau tới từng neighbor, chọn vị trí topology có centrality cao, collude với các attacker khác, hoặc tập trung gây ảnh hưởng vào một vùng của graph. Vì vậy, cùng một tỷ lệ malicious client toàn mạng có thể dẫn đến mức độ tổn thương rất khác nhau tùy vị trí của chúng.

Nghiên cứu của Syros và cộng sự về **Backdoor Attacks in Peer-to-Peer Federated Learning** chỉ ra rằng graph structure có thể được khai thác để lựa chọn malicious nodes, đồng thời đánh giá attack trong nhiều topology, điều kiện non-IID và mức độ hiểu biết khác nhau của attacker về network. Điều này đặc biệt gần với thiết kế thực nghiệm của đề tài hiện tại.

Nguồn quan trọng:

- G. Syros, G. Yar, S. Boboila, C. Nita-Rotaru, A. Oprea, *Backdoor Attacks in Peer-to-Peer Federated Learning*, 2023/2025. https://arxiv.org/abs/2301.09732
- Z. Wang et al., *Defense Strategies Toward Model Poisoning Attacks in Federated Learning: A Survey*, 2022. https://arxiv.org/abs/2202.06414
- Z. Zhang et al., *FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients*, 2022. https://arxiv.org/abs/2207.09209

---

## 4. Các hướng phòng thủ poisoning trong DFL

### 4.1. Local robust aggregation

Một hướng trực tiếp là để mỗi node thay thế averaging thông thường bằng một phép aggregation bền vững hơn. Ý tưởng chung là nhận các update từ neighbor, đo độ bất thường và giảm hoặc loại bỏ ảnh hưởng của update bị nghi ngờ.

**BALANCE** là một baseline rất quan trọng cho đề tài. Fang và cộng sự đề xuất cơ chế trong đó mỗi client dùng chính local model của mình làm reference để đánh giá similarity của model nhận từ neighbor. Điểm mạnh của BALANCE là được thiết kế riêng cho DFL, không cần central server, có convergence guarantee trong các thiết lập strongly convex và non-convex, và đánh đổi tương đối tốt giữa robustness và communication efficiency.

Tuy nhiên, dạng defense này chủ yếu vận hành theo một rule cố định dựa trên similarity. Nó không trực tiếp mô hình hóa attacker như một tác nhân có thể thay đổi chiến lược theo defense mà nó quan sát được.

Nguồn:

- M. Fang et al., *Byzantine-Robust Decentralized Federated Learning*, ACM CCS 2024, pp. 2874-2888. DOI: 10.1145/3658644.3670307. Preprint: https://arxiv.org/abs/2406.10416

### 4.2. Trust và clustering giữa neighbor

Sun và cộng sự đề xuất **DFL-Dual**, kết hợp khoảng cách ở data domain và model domain để nhận diện khác biệt giữa client, sau đó sử dụng trust evaluation và trust bootstrapping để loại Byzantine clients khỏi local aggregation. Một đóng góp đáng chú ý là phương pháp được thiết kế cho tình huống một benign node có thể có hơn một nửa neighbor là Byzantine, vốn là trường hợp khó đối với nhiều robust aggregation rule truyền thống.

DFL-Dual đưa yếu tố trust vào decision cục bộ, vì vậy đây là baseline gần hơn với đề tài game-theoretic so với các aggregator thuần thống kê. Tuy nhiên trust vẫn được cập nhật theo cơ chế thiết kế trước, chưa phải kết quả của một mô hình strategic interaction giữa attacker và defender.

Nguồn:

- P. Sun, X. Liu, Z. Wang, B. Liu, *Byzantine-robust Decentralized Federated Learning via Dual-domain Clustering and Trust Bootstrapping*, CVPR 2024, pp. 24756-24765. https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Byzantine-robust_Decentralized_Federated_Learning_via_Dual-domain_Clustering_and_Trust_Bootstrapping_CVPR_2024_paper.html

### 4.3. Anomaly detection và information-based defense

**AIDFL** sử dụng conditional entropy và mutual information để phát hiện data poisoning trong DFL. Công trình này nhấn mạnh một vấn đề thực nghiệm quan trọng: nhiều cơ chế phát hiện dựa trên khoảng cách hoặc similarity suy giảm hiệu quả khi dữ liệu non-IID, do update hợp lệ từ các benign clients cũng có thể khác nhau đáng kể.

Đây là lý do đề tài hiện tại nên đánh giá defense dưới cả IID và non-IID. Nếu cơ chế strategic defense dùng trust hoặc similarity mà không tính đến heterogeneity, nó rất dễ biến benign heterogeneity thành false positive.

Nguồn:

- *AIDFL: An Information-Driven Anomaly Detector for Data Poisoning in Decentralized Federated Learning*, IEEE Access, vol. 13, 2025, pp. 50017-50031, DOI: 10.1109/ACCESS.2025.3552168. https://ieeexplore.ieee.org/document/10930423/

### 4.4. Attack-agnostic local filtering

Một nghiên cứu năm 2026 trong *Knowledge-Based Systems* đề xuất iterative filtering cho Byzantine-robust DFL. Mỗi client lặp lại việc tính geometric median trên các gradient từ neighbor, loại các gradient vượt ngưỡng khoảng cách, rồi average tập còn lại. Cơ chế này có ưu điểm không phụ thuộc vào một attack construction cụ thể, nhưng theoretical guarantee vẫn yêu cầu điều kiện majority cục bộ nhất định và threshold phù hợp.

Nghiên cứu này cho thấy frontier của DFL defense vẫn tập trung mạnh vào việc tạo **local filter có guarantee tốt hơn**, trong khi câu hỏi attacker sẽ phản ứng chiến lược ra sao với filter đó vẫn tương đối ít được khai thác.

Nguồn:

- J. Wu et al., *Attack-agnostic robust decentralized federated learning*, Knowledge-Based Systems, 2026, Article 116748. DOI: 10.1016/j.knosys.2026.116748. https://www.sciencedirect.com/science/article/pii/S0950705126014747

---

## 5. Game theory cho poisoning defense trong Federated Learning

Game theory phù hợp với poisoning vì attacker và defender không phải hai thành phần tĩnh. Attacker có thể chọn cường độ attack, thời điểm attack và mức stealth; defender có thể điều chỉnh threshold, trust weight, admission policy hoặc mức giao tiếp. Khi một phía thay đổi chiến lược, utility của phía còn lại cũng thay đổi.

Sagduyu, Erpek và Shi mô hình hóa poisoning attack và defense trong FL như một trò chơi giữa quyết định poisoning của attacker và quyết định admission/rejection của defender. Công trình phân tích các chiến lược và Nash equilibrium, qua đó chứng minh giá trị của việc xem security như một strategic interaction thay vì chỉ là một bài toán anomaly detection.

Tuy nhiên kiến trúc của công trình vẫn có một defender mang tính điều phối, phù hợp hơn với FL có cơ chế admission tập trung. Nó chưa giải quyết bài toán mỗi node DFL phải ra quyết định riêng dựa trên local neighborhood.

Nguồn:

- Y. E. Sagduyu, T. Erpek, Y. Shi, *Securing NextG Systems against Poisoning Attacks on Federated Learning: A Game-Theoretic Solution*, 2023. https://arxiv.org/abs/2312.17164

Một hướng mới hơn là **Meta Stackelberg Game**, mô hình hóa adversarial FL dưới Bayesian Stackelberg Markov game để đối phó với mixed và adaptive poisoning attacks. Defender đóng vai trò leader và học policy có khả năng thích nghi khi attack type không chắc chắn hoặc thay đổi. Công trình này rất hữu ích để biện minh cho yếu tố **adaptive attacker** và **incomplete information** trong đề tài.

Dù vậy, mô hình vẫn dựa trên FL có một defender tương đối tập trung và có khả năng xây dựng world model hoặc điều khiển defense policy chung. Việc đưa cùng ý tưởng strategic adaptation vào DFL đòi hỏi xử lý thêm graph topology, local observations, nhiều defender đồng thời và khả năng mỗi malicious node gửi hành vi khác nhau tới từng neighbor.

Nguồn:

- T. Li, H. Li, Y. Pan, T. Xu, Z. Zheng, Q. Zhu, *Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks*, 2024. https://arxiv.org/abs/2410.17431

---

## 6. Research gap

Từ các nhóm nghiên cứu trên có thể rút ra bốn khoảng trống chính.

### Gap 1: DFL defense hiện tại chủ yếu là rule-based local filtering

BALANCE, DFL-Dual, AIDFL và các phương pháp iterative filtering đều cải thiện khả năng chống Byzantine/poisoning bằng cách đánh giá update hoặc neighbor tại từng node. Tuy nhiên phần lớn defense dùng rule được thiết kế trước. Attacker thường được xem là một nguồn update độc hại theo một attack algorithm cố định trong thí nghiệm.

Điều này chưa phản ánh đầy đủ attacker có khả năng **thay đổi cường độ, thời điểm, target và mức stealth khi biết hoặc suy đoán defense**.

### Gap 2: Game-theoretic poisoning defense chủ yếu được nghiên cứu trong FL có cơ chế điều phối tập trung

Các công trình game-theoretic hiện có cho thấy Nash/Stackelberg framework có thể mô hình hóa attacker-defender adaptation. Tuy nhiên defender thường có vai trò global hoặc server-like: quyết định admission, chọn defense policy hoặc quan sát trạng thái hệ thống ở mức rộng.

Trong DFL, không có một defender duy nhất. Mỗi benign node đồng thời là learner, aggregator và defender của neighborhood riêng. Do đó cần một **networked strategic game** nơi utility và action của một agent phụ thuộc vào neighbor và trạng thái local.

### Gap 3: Topology thường được dùng như điều kiện thực nghiệm, chưa được đưa trực tiếp vào strategic utility

Các nghiên cứu P2PFL cho thấy vị trí attacker trong graph ảnh hưởng mạnh đến khả năng lan truyền backdoor. Tuy nhiên trong nhiều defense, topology chỉ là môi trường chạy thuật toán, chưa phải thành phần của action/utility.

Một game model có thể đưa các yếu tố như degree, centrality, số malicious neighbor, propagation risk hoặc communication frequency vào payoff. Đây là điểm giúp nối game theory với đặc trưng riêng của DFL.

### Gap 4: Cần đánh giá đồng thời robustness, clean utility và defense cost

Một defense quá mạnh có thể loại nhầm benign update trong non-IID setting hoặc tăng communication/computation đáng kể. Vì vậy utility của defender không nên chỉ tối ưu model accuracy dưới attack. Nó cần phản ánh cả:

- local/global task loss;
- attack success hoặc mức lan truyền malicious influence;
- false-positive cost khi giảm trust nhầm benign neighbor;
- communication cost;
- computation cost;
- cost khi thay đổi topology hoặc giảm communication frequency.

Đây chính là điểm game theory có lợi: các mục tiêu xung đột có thể được biểu diễn trực tiếp trong utility function.

---

## 7. Hướng mô hình hóa phù hợp cho đề tài

Một one-shot normal-form game có thể dùng để giới thiệu ý tưởng, nhưng chưa đủ mạnh cho DFL vì training diễn ra qua nhiều round và trust thay đổi theo lịch sử. Hướng hợp lý hơn là **repeated Bayesian network game** hoặc **Stackelberg-style repeated game**.

Xét graph giao tiếp \(G=(V,E)\). Mỗi agent \(i\) chỉ quan sát neighborhood \(N_i\).

### Malicious agent

Action có thể gồm:

- attack intensity \(\alpha_i^t\);
- attack timing;
- target neighbor;
- attack type;
- mức stealth hoặc scaling của malicious update.

Utility mẫu:

\[
U_i^A = \lambda_1 D_t + \lambda_2 P_t - \lambda_3 C_i^{attack} - \lambda_4 R_i^{detect}
\]

Trong đó \(D_t\) là model degradation, \(P_t\) là propagation impact, \(C_i^{attack}\) là chi phí attack và \(R_i^{detect}\) là rủi ro bị phát hiện hoặc cô lập.

### Benign agent

Action có thể gồm:

- accept/reject neighbor update;
- trust weight;
- aggregation weight;
- threshold lọc;
- tạm cô lập neighbor;
- tăng/giảm communication frequency.

Utility mẫu:

\[
U_i^D = -\beta_1 L_i - \beta_2 I_i^{poison} - \beta_3 C_i^{comm} - \beta_4 C_i^{comp} - \beta_5 F_i^{FP}
\]

Trong đó \(L_i\) là task loss, \(I_i^{poison}\) là mức ảnh hưởng poisoning, \(C_i^{comm}\) và \(C_i^{comp}\) là chi phí, còn \(F_i^{FP}\) là penalty do loại nhầm benign neighbor.

Nếu attacker không biết hoàn toàn defense và defender không biết node nào malicious, Bayesian formulation phù hợp hơn complete-information game. Nếu muốn mô hình hóa attacker chủ động quan sát defense rồi phản ứng, Stackelberg formulation phù hợp. Nếu cả hai liên tục điều chỉnh qua nhiều round, repeated hoặc Markov game sẽ sát với quá trình DFL hơn.

---

## 8. Research questions đề xuất

Literature review có thể kết thúc bằng các research question sau:

**RQ1.** Làm thế nào mô hình hóa tương tác giữa malicious và benign agents trong DFL như một strategic game khi mỗi agent chỉ có local information về neighbor?

**RQ2.** Một defense policy dựa trên equilibrium hoặc best-response có hạn chế tốt hơn sự lan truyền của malicious updates so với local robust aggregation hoặc trust rule cố định hay không?

**RQ3.** Topology và vị trí của malicious agents ảnh hưởng thế nào tới equilibrium strategy và robustness của hệ thống?

**RQ4.** Defense còn hiệu quả đến mức nào khi attacker có knowledge khác nhau về defense và có khả năng adaptive attack?

**RQ5.** Robustness đạt được phải đánh đổi bao nhiêu clean accuracy, communication cost và computation cost?

---

## 9. Mapping từ literature sang thiết kế thực nghiệm

| Thành phần thí nghiệm | Lý do từ literature | Gợi ý |
|---|---|---|
| Tỷ lệ malicious agents | Robustness phụ thuộc mật độ attacker, đặc biệt theo local neighborhood | 0%, 10%, 20%, 30%, 40% |
| Vị trí attacker | P2PFL attack phụ thuộc graph structure | random vs high-degree/high-centrality |
| Gradient manipulation | Model poisoning trực tiếp | sign flip, scaling, Gaussian/optimized update |
| Label flipping | Data poisoning stress case | ghi rõ đây không phải model poisoning thuần túy |
| Backdoor | Targeted attack và stealth | báo cáo cả clean accuracy và ASR |
| IID / non-IID | Similarity-based defense dễ false positive khi heterogeneous | Dirichlet partition với nhiều mức alpha |
| Topology | Ảnh hưởng propagation và local malicious ratio | small-world, scale-free, random geometric |
| Attacker knowledge | Kiểm tra adaptive robustness | oblivious, partial-knowledge, defense-aware |
| Baseline DFL | Đo thiệt hại khi không defense | plain neighbor averaging |
| BALANCE | Byzantine-robust DFL baseline mạnh | bắt buộc nên có |
| DFL-Dual | Trust/clustering baseline | rất phù hợp để so sánh |
| Reputation/local filter | Gần với action trust-weight của đề tài | chọn 1-2 baseline đơn giản, tái hiện được |
| Clean run | Defense không được tự phá mô hình | accuracy/loss khi không attack |
| Communication cost | Strategic policy có thể thay đổi tần suất/neighbor | bytes/messages per round |
| Computation cost | Một số defense tốn nhiều local scoring | wall-clock hoặc relative overhead |

---

## 10. Bộ paper lõi nên đọc kỹ trước khi viết bản chính thức

### Nhóm A - DFL nền tảng

1. Lalitha et al. (2019), *Peer-to-peer Federated Learning on Graphs*.
2. Guha Roy et al. (2019), *BrainTorrent*.
3. Hallaji et al. (2024), *Decentralized Federated Learning: A Survey on Security and Privacy*.

### Nhóm B - Attack trong DFL/P2PFL

4. Syros et al. (2023/2025), *Backdoor Attacks in Peer-to-Peer Federated Learning*.
5. Wang et al. (2022), *Defense Strategies Toward Model Poisoning Attacks in Federated Learning: A Survey*.

### Nhóm C - Defense DFL

6. Fang et al. (CCS 2024), *Byzantine-Robust Decentralized Federated Learning* / BALANCE.
7. Sun et al. (CVPR 2024), *DFL-Dual*.
8. AIDFL (IEEE Access 2025).
9. Wu et al. (Knowledge-Based Systems 2026), *Attack-agnostic robust decentralized federated learning*.

### Nhóm D - Game-theoretic poisoning defense

10. Sagduyu et al. (2023), *Securing NextG Systems against Poisoning Attacks on Federated Learning: A Game-Theoretic Solution*.
11. Li et al. (2024), *Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks*.

### Nhóm E - Centralized FL baselines để làm rõ giới hạn khi chuyển sang DFL

12. Zhang et al. (2022), *FLDetector*.
13. Các robust aggregation kinh điển như Krum, coordinate-wise median và trimmed mean chỉ nên dùng để tạo bối cảnh; không nên biến literature review thành lịch sử toàn bộ Byzantine SGD.

---

## 11. Đoạn tổng hợp có thể dùng làm kết thúc Literature Review

Các nghiên cứu hiện tại đã xác nhận rằng việc loại bỏ central server không làm biến mất poisoning threat mà còn chuyển trách nhiệm phòng thủ xuống từng client trong DFL. Những phương pháp như BALANCE, DFL-Dual, AIDFL và iterative filtering cho thấy local similarity, trust, information-based anomaly detection và robust statistics có thể hạn chế ảnh hưởng của Byzantine clients. Tuy nhiên các cơ chế này phần lớn xem defense như một rule cố định và đánh giá attacker thông qua một tập attack strategy đã định nghĩa trước. Song song với đó, các nghiên cứu game-theoretic trong FL cho thấy attacker-defender interaction có thể được mô hình hóa thông qua Nash hoặc Stackelberg equilibrium để xử lý strategic và adaptive attacks, nhưng các mô hình này chủ yếu giả định một defender có vai trò điều phối tương tự server. Do đó vẫn tồn tại khoảng trống trong việc thiết kế một defense cho DFL mà mỗi benign agent đưa ra quyết định dựa trên local observations, topology và lịch sử tương tác, đồng thời dự đoán phản ứng chiến lược của malicious neighbors. Đề tài này hướng tới khoảng trống đó bằng cách mô hình hóa DFL như một strategic game trên graph và sử dụng kết quả phân tích game để điều chỉnh filtering, trust, neighbor selection hoặc communication policy theo cách phân tán.

