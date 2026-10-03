# Mô hình trò chơi cho poisoning trong Decentralized Federated Learning

> Ghi chú thiết kế nghiên cứu, dựa trên `literature_review.md`, `game_theory_research_note.md` và thảo luận về repeated Bayesian, Markov và Stackelberg game. Các công thức dưới đây là đề xuất mô hình hóa, chưa phải thuật toán hoặc kết quả equilibrium đã được chứng minh.

## 1. Kết luận chính

- DFL vẫn có thể được mô hình hóa bằng attacker–defender game, nhưng defender là các benign receiving nodes, không phải một central server.
- Markov game cần một trạng thái đủ để mô tả transition; **không yêu cầu có một global model**.
- Trạng thái toàn mạng là mô tả toán học của hệ thống. Không có nghĩa mọi node được truy cập mô tả đó.
- Mỗi node lựa chọn action dựa trên local observations và lịch sử tương tác của chính mình.
- Attacker gửi update trước, defender phản ứng sau chỉ xác định thứ tự hành động. Điều đó chưa tự động biện minh cho Stackelberg game.

Hướng khởi đầu đề xuất là **repeated Bayesian network game**, sau đó bổ sung state transitions và xem xét một **partially observable networked stochastic game with private node types**. Chỉ thêm Stackelberg khi có giả định policy commitment rõ ràng.

## 2. Ba khái niệm không nên đồng nhất

| Khái niệm | Ý nghĩa trong DFL | Node có truy cập không? |
|---|---|---|
| Local model \(w_i^t\) | Model hiện tại của node \(i\) | Có, với model của chính nó |
| Joint network state \(S_t\) | Trạng thái chung của toàn bộ local models, trust, topology và các biến động học cần thiết | Thường không |
| Global model \(\bar w_t\) | Một model chung được duy trì bởi protocol, hoặc một model tổng hợp do evaluator xây dựng | Không mặc nhiên tồn tại hoặc được cung cấp |

Một Markov game có thể có \(S_t\) nhưng không có \(\bar w_t\). Từ “global” trong global state nói về phạm vi mô tả, không nói về quyền truy cập hay kiến trúc điều phối.

DFL có thể hướng tới đồng thuận hoặc các model gần nhau theo thời gian. Điều đó không có nghĩa ở mọi round tồn tại một global model mà các node đều biết.

## 3. Players và thông tin riêng

Xét graph giao tiếp \(G_t=(V,E_t)\). Đặt \(N_i^t\) là tập các node có thể gửi thông điệp đến node \(i\) tại round \(t\).

### Benign players

Mỗi benign node là một learner, sender, local aggregator và defender. Nó có thể lựa chọn aggregation weights cho các neighbor, hoặc mở rộng sang reject, isolation và communication decisions nếu protocol cho phép.

Các benign nodes có thể dùng chung một policy architecture nhưng thực thi độc lập trên local observations. Shared policy không đồng nghĩa centralized defender.

### Attacker player

Nếu một attacker kiểm soát tập node \(M\subseteq V\), có thể mô hình attacker là **một decision-maker** lựa chọn joint attack actions của các node trong \(M\). Không nhất thiết coi từng compromised node là một player độc lập.

Action có thể gồm attack timing, intensity, data poisoning, model manipulation và recipient-specific messages. Nếu protocol yêu cầu broadcast một thông điệp giống nhau cho mọi neighbor, action space phải tuân theo ràng buộc đó.

Không mặc nhiên cho attacker biết toàn bộ graph hoặc state. Attacker knowledge là một phần riêng của threat model.

### Hidden type không phải model contamination

Một định nghĩa ban đầu đơn giản là:

\[
\theta_j\in\{\text{benign},\text{malicious}\}.
\]

Node benign có thể nhận poisoned update rồi gửi một model đã bị ảnh hưởng sang neighbor khác. Nó vẫn benign về ý định. Vì vậy cần phân biệt:

- **Type:** có chủ đích tấn công hay không.
- **Action:** đang gửi trung thực, im lặng hay thực hiện poisoning.
- **Model contamination:** model hiện tại chịu ảnh hưởng poisoning đến mức nào.

Nếu attacker có thể compromise thêm node, cần đưa trạng thái kiểm soát node vào transition. Nếu không, tập compromised nodes có thể cố định xuyên suốt thí nghiệm.

## 4. Trạng thái Markov ở mức toàn mạng

Một joint network state minh họa là:

\[
S_t=\left(G_t,\{w_i^t,z_i^t\}_{i\in V},
\{\tau_{ij}^t,m_{ij}^t\}_{(j,i)\in E_t},\xi_t^A\right).
\]

Trong đó:

- \(w_i^t\): local model parameters.
- \(z_i^t\): optimizer/local training state, nếu cần cho transition.
- \(\tau_{ij}^t\): trust mà receiver \(i\) dành cho sender \(j\); trust có hướng và không nhất thiết đối xứng.
- \(m_{ij}^t\): bộ nhớ cần cho scoring và trust update.
- \(\xi_t^A\): attacker budget, trạng thái kiểm soát node hoặc bộ nhớ thích nghi, nếu chúng ảnh hưởng bước tiếp theo.

Private types có thể được đưa vào hidden state hoặc xem là tham số cố định của transition. Các data partitions cố định là tham số môi trường. Nếu dữ liệu, learning rate hoặc protocol thay đổi theo lịch thời gian, phải biểu diễn biến cần thiết trong state hoặc dùng time-dependent transition.

Markov property yêu cầu state đủ để xác định phân phối bước tiếp theo khi biết joint actions:

\[
\Pr(S_{t+1}\mid S_{0:t},\mathbf a_{0:t})
=\Pr(S_{t+1}\mid S_t,\mathbf a_t),
\]

với các tham số môi trường/type cố định được hiểu là đã điều kiện hóa.

State trên chỉ là template: cần bổ sung hoặc lược bỏ biến theo protocol thực tế. Không cần lưu toàn bộ lịch sử nếu thuật toán chỉ sử dụng một bộ nhớ đủ đã được xác định.

**Accuracy, trust và similarity chưa mặc nhiên là một exact Markov state.** Hai mạng có cùng các summary metrics vẫn có thể có parameters khác nhau và chuyển sang trạng thái khác nhau sau cùng một action. Nếu dùng summary features, nên gọi là state abstraction hoặc approximate Markov representation.

## 5. Transition được tạo bởi các bước cục bộ

Ví dụ dưới đây dùng synchronous rounds và model exchange. Gradient/delta exchange cần công thức aggregation tương ứng. Với asynchronous DFL, có thể phải thêm message queues, delays hoặc model versions vào state.

### 5.1. Local training và thông điệp gửi đi

Node \(j\) thực hiện local training:

\[
\widetilde w_j^t=\operatorname{LocalTrain}_j(w_j^t,D_j,z_j^t).
\]

Optimizer state cũng được cập nhật theo training rule. Thông điệp gửi đến \(i\) là:

\[
u_{j\rightarrow i}^t=
\begin{cases}
\widetilde w_j^t,&\text{gửi trung thực},\\
\operatorname{Manipulate}_{j\rightarrow i}(\widetilde w_j^t,a_j^t),&\text{model poisoning}.
\end{cases}
\]

Với data poisoning, attacker thay đổi dữ liệu/quá trình local training trước khi tạo thông điệp. Với silent action, không có thông điệp và tập sender khả dụng thay đổi tương ứng. Sender trung thực vẫn có thể gửi model đã chịu ảnh hưởng poisoning từ round trước.

### 5.2. Defense action và local aggregation

Đặt \(R_i^t\subseteq N_i^t\) là tập neighbor có thông điệp khả dụng. Node \(i\) chọn:

\[
d_i^t=\{\alpha_{ij}^t:j\in R_i^t\cup\{i\}\},
\qquad \alpha_{ij}^t\geq0,
\qquad \sum_{j\in R_i^t\cup\{i\}}\alpha_{ij}^t=1.
\]

Reject tương ứng weight bằng 0; downweight là giảm weight so với mức tham chiếu; accept là sử dụng weight được policy cho phép, không nhất thiết bằng 1.

\[
w_i^{t+1}=\alpha_{ii}^t\widetilde w_i^t
+\sum_{j\in R_i^t}\alpha_{ij}^t u_{j\rightarrow i}^t.
\]

Nếu reject toàn bộ neighbor, node có thể tiếp tục bằng local model với self-weight bằng 1. Điều này không nhất thiết dừng local learning, nhưng làm mất lợi ích trao đổi dữ liệu/model và có thể giảm chất lượng hoặc tốc độ hội tụ.

### 5.3. Trust, memory và topology

Node cập nhật trust/bộ nhớ theo rule đã xác định, sử dụng evidence thực sự quan sát được. Có thể cập nhật effective communication graph nếu isolation là một action được phép.

Ghép các bước cục bộ tạo ra:

\[
S_{t+1}\sim P(\cdot\mid S_t,\mathbf a_t^A,\mathbf d_t^D,\theta).
\]

Không có server thực hiện joint transition. Nó là mô tả chung của các bước phân tán. Poisoning có thể lan nhiều hop khi local model bị ảnh hưởng tiếp tục được gửi đi ở các round sau.

## 6. Local nodes thực sự dùng thông tin nào?

### 6.1. Không lấy thông tin từ một global model giả định

Mỗi node dùng model của chính mình và thông điệp được protocol chuyển đến nó. Một local observation tại thời điểm chọn defense weights có thể là:

\[
o_i^t=\left(w_i^t,\widetilde w_i^t,
\{u_{j\rightarrow i}^t,\tau_{ij}^t\}_{j\in R_i^t},
N_i^t,\text{local validation signals}\right).
\]

Các features tính được cục bộ có thể gồm norm, similarity với local model/các thông điệp nhận được, lịch sử neighbor, và loss trên local validation data nếu có. Similarity thấp không đồng nghĩa malicious, đặc biệt trong non-IID setting.

Local validation cũng không mặc nhiên phát hiện được backdoor: node có thể không có trigger hoặc target samples cần thiết.

### 6.2. Policy dùng local history hoặc local memory

\[
d_i^t\sim\pi_i^D(\cdot\mid h_i^t),
\]

trong đó \(h_i^t\) gồm các observations và actions đã biết của node \(i\), đến thời điểm quyết định hiện tại.

Trong thực nghiệm có thể rút gọn thành:

\[
x_i^t=\phi_i(h_i^t),\qquad
d_i^t\sim\pi_i^D(\cdot\mid x_i^t).
\]

Ví dụ \(x_i^t\) chứa trust vector, recent scores, local loss, available neighbors và recurrent memory. Đây là policy input, không mặc nhiên là exact Markov state.

**Toàn mạng Markov không bảo đảm local observation Markov.** Node ngoài neighborhood có thể thay đổi model của neighbor, khiến phân phối thông điệp tương lai khác nhau dù local features hiện tại giống nhau.

### 6.3. Bayesian belief

Node có thể duy trì:

\[
b_{ij}^t=\Pr(\theta_j=\text{malicious}\mid h_i^t).
\]

Bayesian update cần likelihood/observation model phù hợp, không chỉ một công thức cộng trừ trust. Trust score chưa được calibrated không nên được gọi trực tiếp là xác suất malicious.

Belief về type neighbor không nhất thiết là sufficient statistic cho toàn bộ hidden network state. Trong multi-agent setting, diễn biến còn phụ thuộc policies và private information của các node khác. Không nên khẳng định một vector type beliefs tự động biến local problem thành fully observed Markov game.

## 7. Nếu evaluator xây dựng một “global model” thì sao?

Evaluator có thể tính model trung bình:

\[
\bar w_t=\frac{1}{|V|}\sum_{i\in V}w_i^t.
\]

Đây là lựa chọn đánh giá, không phải thành phần bắt buộc của DFL hoặc Markov game. Với personalized/heterogeneous models, model averaging có thể không phù hợp; có thể báo cáo mean, worst-node và phân phối performance của các local models thay thế.

Trong strict local-information DFL:

- \(\bar w_t\), global malicious identities, global ASR và toàn bộ topology có thể được evaluator sử dụng để phân tích.
- Không đưa các đại lượng đó vào local policy input nếu node không được protocol cung cấp.
- Một global metric dùng để báo cáo kết quả không phải là một observation của node.

Nếu muốn node sử dụng thông tin tổng hợp toàn mạng, phải bổ sung cơ chế communication như gossip/consensus hoặc một protocol tổng hợp khác. Khi đó node nhận một estimate có độ trễ/sai số, không mặc nhiên biết exact global state; cần tính communication cost và nguy cơ manipulation. Đây là mở rộng scope, không phải điều kiện để dùng Markov game.

## 8. Utility, reward và khả năng quan sát

Phân biệt **objective dùng để mô hình hóa preferences**, **feedback node nhận được**, và **metric evaluator báo cáo**.

Một local reward có thể tính được, nếu node có local validation set, là:

\[
r_i^t=L_i^{val}(w_i^t)-L_i^{val}(w_i^{t+1})
-\lambda_c C_i^{comm,t}-\lambda_p C_i^{comp,t}.
\]

Reward này phản ánh local improvement và chi phí, nhưng không đảm bảo đo được backdoor damage, global convergence hay đóng góp riêng của từng neighbor. Các proxy bổ sung cần được định nghĩa và đánh giá rõ.

Attacker có thể tối ưu damage/ASR trên mục tiêu và dữ liệu nó truy cập được, trừ attack effort. Giá trị của duy trì trust có thể xuất hiện thông qua future attack opportunities trong discounted utility, không nhất thiết cần thêm một trust bonus riêng.

\[
J_i=\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_i^t\right].
\]

True false-positive rate, contamination và ASR có thể là latent/evaluation quantities. Không dùng chúng như feedback quan sát trực tiếp nếu threat model không cho phép. Nếu dùng privileged simulator information để huấn luyện, phải nói rõ đó là **centralized training with decentralized execution**, không phải fully local online learning. Đây là lựa chọn riêng, không phải yêu cầu của mô hình.

## 9. Quan hệ giữa các local games

Tương tác giữa receiver \(i\) và sender \(j\) có thể được phân tích như một local subgame, nhưng các edge games không độc lập:

- Weights của mọi neighbor cùng chịu ràng buộc chuẩn hóa.
- Các updates được aggregation cùng nhau, có tương tác về tác động.
- Local model sau aggregation ảnh hưởng các thông điệp gửi ở round sau.
- Topology quyết định các đường lan truyền và cơ hội tương tác tiếp theo.

Vì vậy, mô hình hai-player cục bộ là baseline/approximation, không nên tuyên bố equilibrium của từng cạnh tự động là equilibrium toàn mạng.

## 10. Khi nào dùng Stackelberg?

Timeline hiện tại:

\[
\text{gửi update}\rightarrow\text{quan sát}\rightarrow\text{chọn defense weight}
\]

trước hết là sequential interaction with incomplete information. Với lịch sử, beliefs và sequential rationality được định nghĩa đầy đủ, có thể xem xét sequential Bayesian solution concepts; chưa nên khẳng định một equilibrium cụ thể khi chưa có solver/analysis.

Stackelberg phù hợp nếu có policy commitment rõ ràng, ví dụ:

\[
\text{defense policy được cam kết trước}
\rightarrow\text{attacker quan sát/suy đoán policy}
\rightarrow\text{attacker best-responds}.
\]

Trong DFL cần giải thích ai cam kết, cam kết điều gì, follower biết gì, và các benign nodes có phối hợp hay không. Không mặc nhiên biến toàn bộ benign nodes thành một central leader. Nếu defender cũng liên tục thay policy sau khi quan sát attacker, phải mô tả dynamic timing thay vì dùng nhãn Stackelberg cho thứ tự sự kiện đơn thuần.

## 11. Phạm vi triển khai đề xuất

1. Graph cố định và synchronous communication rounds.
2. Một attacker kiểm soát tập node cố định; phân biệt attacker controller với compromised nodes.
3. Benign nodes thực thi local defense policies; không truy cập global state hoặc malicious identities.
4. Attack actions ban đầu gồm honest/silent/poison với một tập intensity hữu hạn; defender chọn reject/downweight/accept qua aggregation weights.
5. Ghi rõ self-weight, missing-message handling, local validation assumptions và việc có cho phép recipient-specific messages hay không.
6. Dùng local trust/history/features làm policy input; không khẳng định exact Markov sufficiency nếu chưa có chứng minh.
7. Giữ attacker hidden types, model contamination và suspicious observations là các khái niệm khác nhau.
8. Đánh giá clean utility, attack impact, false positives, propagation qua nhiều hop và defense costs bằng evaluator tách biệt.
9. Bắt đầu từ repeated Bayesian network game; thêm Markov structure để nghiên cứu hậu quả dài hạn. Chỉ thêm Stackelberg nếu commitment/best-response là research question thực sự.

### Câu mô tả mô hình có thể dùng trong luận văn

> We model poisoning in DFL as a repeated Bayesian interaction on a communication graph. Each benign node selects local aggregation actions using only its observations and interaction history, while an attacker controls the messages of a subset of nodes. The joint evolution of local models, directed trust and communication states defines network-level stochastic dynamics; this state is not assumed to be observable by individual nodes. A Markov abstraction may be used for tractable policy learning, with its information and approximation limits stated explicitly.

**Tóm lại:** local nodes không cần “lấy thông tin từ global model” để tham gia Markov game. Chúng tác động lên một hệ thống có joint state bằng actions dựa trên local information; joint state chỉ mô tả cách toàn mạng cùng tiến triển.
