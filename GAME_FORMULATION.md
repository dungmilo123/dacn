# Context và khung game — Răn đe parameter poisoning bằng trọng số aggregation

Ngày cập nhật: 2026-10-28.

**Trạng thái:** episode-policy game cho thấy opportunity-aware poisoning vẫn có lợi. Đã đề xuất nhánh **audit-backed collateral utility** (§21), với conditional first-deviation deterrence proof và participation/capital constraints; chưa có prover/escrow triển khai hoặc measured operational guarantee. Chia đều revenue vẫn chưa đủ trên frozen trajectories. Historical model/game/robustness evidence giữ nguyên; official test chưa dùng. Kết quả và ưu tiên redesign tại [ROADMAP.md](ROADMAP.md); ví dụ mục7 vẫn chỉ minh họa.

**Phạm vi:** mục tiêu chống đầu độc mô hình trong [topics.md](topics.md). Tài liệu này đặc tả game; [ROADMAP.md](ROADMAP.md) ghi cấu hình đã chạy, kết quả pilot và việc tiếp theo. Utility thị phần thay cho utility trừ trực tiếp accuracy đối thủ trong bản trước; không dùng song song hai utility để chọn kết quả mong muốn.

## 1. Context và câu hỏi nghiên cứu

Các node là những tổ chức cung cấp cùng loại dịch vụ dự đoán, dùng cùng không gian nhãn và kiến trúc mô hình tương thích để trao đổi tham số. Mỗi tổ chức có dữ liệu riêng, model riêng và cạnh tranh khách hàng. Học liên kết phi tập trung có thể giúp cải thiện model, nhưng một node cũng có thể muốn làm model đối thủ kém đi để thu hút khách hàng. Không có server trung tâm aggregate hoặc biết nhãn A/D thật.

Node có thể giữ bản model riêng không bị thao túng trực tiếp, trong khi gửi tham số bị đầu độc cho neighbor. Poisoning có cost và có thể làm hỏng nguồn update hữu ích quay lại trong tương lai. Không gán identity attacker/defender bất biến; hành vi A/D thay đổi theo quyết định.

- **IID:** dữ liệu train cục bộ lấy mẫu từ cùng phân phối; không đồng nghĩa mọi mẫu, minibatch và trajectory giống nhau.
- **Non-IID:** phân phối train khác nhau. Update hợp lệ khác biệt có thể bị nhầm với poisoning. Phân phối train của node không tự động đồng nhất với phân bố khách hàng thị trường.

Non-IID không tự sinh động cơ tấn công. Động cơ đến từ cạnh tranh khách hàng. Lợi ích của học chung phải được kiểm tra bằng đối chứng tự học, không giả định trước.

### Câu hỏi nghiên cứu

> Trong DFL giữa các node cạnh tranh khách hàng, có thể thiết kế quy tắc aggregation weight chỉ dùng observation cục bộ để parameter poisoning mất lợi thế, hạn chế sự tồn tại/xâm nhập của hành vi này và duy trì chất lượng model trên toàn mạng hay không; trong những điều kiện IID/non-IID nào?

Hai yêu cầu độc lập:

1. **Răn đe:** poisoning trở nên kém có lợi theo payoff/return và giảm qua quá trình thích nghi.
2. **Bảo vệ:** chất lượng học dưới tấn công được duy trì, với clean degradation và chi phí được báo rõ.

Giảm tỷ lệ A, tăng thị phần, all-D hoặc một equilibrium ổn định chưa tự chứng minh yêu cầu thứ hai.

## 2. Quyết định và trạng thái thiết kế

| Thành phần | Nội dung | Trạng thái |
| --- | --- | --- |
| Player | Node có identity xuyên suốt; không chia phe cố định | Đã thống nhất |
| Action | Mỗi lượt chọn đúng một trong A/D | Đã thống nhất |
| Attack | Untargeted amplified sign-flip local update, scale=5; giữ model local sạch | Đã chạy pilot; công thức tại mục 10 |
| Defense | W0/W1 validation và conditional access bằng EMA/binary/magnitude ledger | Đã chạy đối chứng; chưa chọn final defense |
| Utility | Chung dạng hàm, tham số có thể khác; nhất quán cho A và D | Đã thống nhất |
| Cạnh tranh | Tệp khách hàng tiềm năng cố định; khách hàng đổi dịch vụ | Bối cảnh đã chọn |
| Chất lượng kỳ vọng | tau_t=min(0.8,0.1+0.05t), ngoại sinh, không lấy average accuracy | Cấu hình pilot; cần sensitivity |
| Outside option | Khách hàng có thể không dùng node nào; expected share trên tổng khách hàng tiềm năng | Đã dùng trong pilot |
| Timing | Nhận message cũ, aggregate/quan sát rồi chọn action tạo message tiếp theo | Đã thảo luận; đặc tả tại mục 6 |
| Memory | Baseline đầu không history; policy §18 dùng own link-state, receipt/action trước và own share | Đã chạy local finite-feature learner, không full history window |
| Cost | Cả hai action có cost, có thể đổi theo state/time; pilot cố định c_D=0.005, c_A=0.006 | Giả định utility units; chưa đo/calibrate chi phí kinh tế |
| Dynamics | EGT là trục phân tích frozen game; bổ sung episodic policy-gradient baseline §18 | PG không phải replicator/EGT; chưa chứng minh hội tụ |
| Horizon | Pilot T20; learner sensitivity T10/20/25 với settlement cuối; oracle H1–3, gamma=.95 | Đã kiểm finite-horizon effects; không infinite-horizon hoặc dynamic Nash |

## 3. Players, thị trường và thông tin riêng

Đặt N={1,...,n}, n>=2, trên đồ thị G=(N,E) cố định. Tập neighbor nhận của i là N_i. Đồ thị giao tiếp không cần trùng cấu trúc cạnh tranh thị trường. Mỗi node có train/validation riêng, model riêng và trạng thái optimizer cần thiết.

![Lattice 4x4, 24 cạnh vô hướng, message đi cả hai chiều trên mỗi cạnh](figures/lattice_4x4.svg)

Hình trên là instance 4x4 row-major dùng cho pilot: |N|=16, |E|=24 cạnh von Neumann, mỗi cạnh mang message hai chiều nên có 48 arc có hướng; bậc là 2/3/4 cho góc/cạnh/trong. Hình chưa gán attacker, defender hay action — chỉ topology. Sinh lại bằng `python experiments/viz_lattice.py static`.

Khi chạy pilot, `--viz` mở trang localhost vẽ topology thật của run và cập nhật mỗi round: node viền đứt đỏ = message bị đầu độc, cạnh xanh = message được nhận, nhãn dưới node = share; panel phải giữ τ/q_net/q_min/outside cùng bảng quality, share, payoff, p(A), số neighbor nhận của từng node. `--viz-step` chặn sau mỗi round cho tới khi bấm "round tiếp theo"; nút ◀ ▶ và slider xem lại các round đã chạy. Topology lấy từ `--topology ring|lattice` (`lattice` cần số node là số chính phương).

Có M khách hàng tiềm năng trong episode, không thêm/bớt khách hàng. M được công bố; khách hàng có thể đổi node hoặc không sử dụng dịch vụ trong từng lượt. Không cố định tệp phục vụ của từng node.

Có thể chia thị trường thành các nhóm g với tỷ trọng mu_g>=0, tổng bằng 1, cố định trước thí nghiệm. Baseline bắt đầu một nhóm; non-IID có thể xét nhiều nhóm. Mọi nhóm có cùng tập nhà cung cấp trong mô hình khởi đầu; chưa cho node tự chọn chỉ phục vụ nhóm dễ.

Ba loại dữ liệu phải tách:

- **Train:** dùng local training.
- **Local validation:** dùng cho observation/weight; không phải nhãn chất lượng thật của đối thủ.
- **Market feedback và final test:** phản hồi thị trường đến từ chất lượng dịch vụ trên dữ liệu/nhu cầu thị trường, không dùng final test. Final test được giữ riêng, không tạo market share, reward hoặc policy input.

Simulator có thể mô phỏng phản ứng thị trường từ quality của model thật đang phục vụ. Đây là giả định về môi trường khách hàng, không có nghĩa mọi agent được cấp quality đó. Chưa tuyên bố đã có mô hình thị trường được kiểm định thực tế.

## 4. Thị trường, quality và utility

### 4.1. Quality và kỳ vọng ngoại sinh

Gọi q_ig^t trong [0,1] là quality dịch vụ của model riêng w_i^t đối với nhóm g tại lượt t; lớn hơn tốt hơn. Trong thị trường mô phỏng, quality này được xác định trên nguồn đánh giá thị trường riêng, không trên final test.

Node không biết q_jg thật của đối thủ. Nó có thể tính accuracy của message m_j trên validation riêng; đó chỉ là quality của **bản được gửi** trên phân phối của receiver, không phải quality của model riêng đối thủ.

Đặt tau_t trong [0,1] là mức chất lượng khách hàng kỳ vọng. Đề xuất:

\[
\tau_t=\min(\tau_{\max},\tau_0+v t),\qquad v\geq0.
\]

Lịch công khai, ngoại sinh và cố định trước chạy final; không phụ thuộc average quality, tỷ lệ A hoặc cơ chế W. Chọn mức khả thi bằng pilot clean learning và có đối chứng v=0. Không tăng kỳ vọng vô hạn hoặc chọn lịch chỉ để ép mọi dịch vụ mất khách.

### 4.2. Quy tắc lựa chọn có outside option

Một đặc tả mềm đề xuất cho nhóm g:

\[
s_{ig}^t=
\frac{\exp[\kappa(q_{ig}^t-\tau_t)]}
{1+\sum_{j\in N}\exp[\kappa(q_{jg}^t-\tau_t)]},
\qquad
s_{0g}^t=\frac{1}{1+\sum_{j\in N}\exp[\kappa(q_{jg}^t-\tau_t)]},
\quad \kappa>0.
\]

\[
\sigma_i^t=\sum_g\mu_g s_{ig}^t,
\qquad \sigma_0^t=\sum_g\mu_g s_{0g}^t,
\qquad \sigma_0^t+\sum_i\sigma_i^t=1.
\]

Đây là expected share; M*sigma_i không cần là số nguyên. Baseline dùng expected share để tách ngẫu nhiên thị trường khỏi ngẫu nhiên học. Nếu thêm hữu hạn khách hàng lấy mẫu lựa chọn, số đếm phải bảo toàn tổng M cả outside option và phải phân biệt realized với expected share.

Mức tau là kỳ vọng mềm, không phải cutoff: q<tau vẫn có thể còn khách hàng. Tăng quality riêng làm share riêng tăng khi các yếu tố khác cố định; tăng tau đồng loạt làm nhu cầu sang outside option tăng. Chỉ trừ tau trong softmax không có outside option sẽ bị triệt tiêu và không tạo tác dụng.

Mô hình khởi đầu giả định cùng giá, không xét thương hiệu/quảng cáo/cost chuyển dịch vụ, và phản ứng theo quality hiện tại. Đây là đơn giản hóa; không mặc định khách hàng thực tế biết accuracy tức thời chính xác.

### 4.3. Utility chung dạng hàm

Với v_i>0 là giá trị một đơn vị khách hàng, net payoff của action t được quyết toán tại t+1:

\[
u_i^{t+1}=v_i M\sigma_i^{t+1}-c_i^t(s_t,a_i^t,s_{t+1}).
\]

Có thể chuẩn hóa cùng đại lượng theo v_i*M:

\[
\bar u_i^{t+1}=\sigma_i^{t+1}-\bar c_i^t,
\qquad \bar c_i^t=\frac{c_i^t}{v_i M}.
\]

Dùng nhất quán một thang trong mỗi thí nghiệm, không cộng thêm khoản thưởng trực tiếp vì gây damage. Đây là utility riêng của node, không phải shared team reward. v_i và tham số cost có thể khác nhưng giữ cố định về sở thích qua A/D. Baseline cùng hệ số kinh tế để cô lập tác động dữ liệu/topology; heterogeneity là phân tích độ nhạy.

Chất lượng riêng tốt hơn có thể tăng lượng khách phục vụ; làm đối thủ kém đi cũng có thể làm share riêng tăng. Outside option tạo áp lực chất lượng tuyệt đối nhưng **không loại bỏ động cơ phá đối thủ** hoặc bảo đảm tăng quality toàn mạng.

\[
J_i(\pi_i,\pi_{-i};W)
=\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t\bar u_i^{t+1}\right],
\qquad 0\leq\gamma\leq1.
\]

Không còn giải thưởng first-to-threshold hay utility trừ trực tiếp accuracy đối thủ. Node quan sát share riêng nên có feedback liên quan trực tiếp đến utility, không cần được cấp accuracy đối thủ.

## 5. Action, cost và quyền phòng thủ

Mỗi node chọn a_i^t thuộc {D,A}. D gửi model hợp lệ; A gửi bản parameter-manipulated. D là cách gọi defender/cooperative ở lớp gửi message, không phải một bộ phòng thủ mà A bị cấm dùng.

Đề xuất cost thực thi:

\[
c_i^t(D)=b_i^t,
\qquad c_i^t(A)=b_i^t+m_i^t,
\qquad b_i^t>0,\ m_i^t\geq0.
\]

b gồm các bước dùng chung như training/communication/aggregation; m là chi phí thao túng bổ sung nếu thực thi thực tế có nó. Cho phép cost đổi theo state/time nhưng phải quy định cơ chế đổi và thông tin biết trước. Cost thực nhận được phản hồi sau thực thi. Chọn cùng quy luật cost giữa các đối chứng.

Hai cost dương chưa tạo răn đe nếu chênh lệch cost bằng 0. Kiểm tra m tiến về 0; không thêm penalty lớn hoặc time schedule tùy ý để ép chuyển vai. Cost không được biết trước chính xác thì policy chỉ dùng estimate/phân phối được cấp.

### Local-only weight rule

Tại t>=1, nhận message m_j^{t-1}, node i dùng model local \(\ell_i^{t-1}\) được giữ sau training lượt trước và tính:

\[
\omega_i^t=W_i\left(\ell_i^{t-1},\{m_j^{t-1}:j\in\mathcal N_i\},
D_i^{val},\sigma_i^{t-1},t,\tau_t\right).
\]

\[
\omega_{ij}^t\geq0,\qquad
\sum_{j\in\mathcal N_i\cup\{i\}}\omega_{ij}^t=1,
\qquad
w_i^t=\omega_{ii}^t \ell_i^{t-1}
+\sum_{j\in\mathcal N_i}\omega_{ij}^t m_j^{t-1}.
\]

\(\ell_i^{t-1}\) là bản local không bị thao túng trực tiếp, khác với bản gửi \(m_i^{t-1}\). \(w_i^t\) là model phục vụ sau aggregation. Các ký hiệu ell_i và \(\ell_i\) dưới đây cùng chỉ model local.

W dùng share của lượt trước vì share hiện tại chỉ có sau khi aggregate/đánh giá thị trường. Không cho W dùng share sau quyết định weight của chính lượt đó, tránh vòng phụ thuộc. W là quy tắc người nghiên cứu thiết kế, không phải action chiến lược bổ sung.

Weight 0 loại ảnh hưởng một message trong lượt, không xóa cạnh/blacklist node. Nếu không nhận message nào thì self-weight=1. Không áp lower bound dương chỉ để buộc nhận một phần update độc. Nhưng luôn self-weight=1 phải được so với local-only và kiểm tra mất lợi ích học liên kết.

Cả A và D dùng receive-side W như nhau. Miền sai lệch ban đầu chỉ cho thao túng bản gửi, không tùy ý sửa W, training, dữ liệu hay thị phần báo cáo. Không mặc định có thể cưỡng chế node thật tuân thủ; attacker tự thay receive-side protocol nằm ngoài bảo đảm ban đầu.

## 6. Pipeline nhận trước, chọn sau và observation

### 6.1. Timing đồng bộ

Dùng ell_i^{t-1} là model local được giữ từ lượt trước; m_i^{t-1} là bản đã gửi. Với t=1,...,T-1, thứ tự là:

1. Nhận m_j^{t-1}; mọi message này được tạo từ action lượt trước.
2. Tính W từ model local ell_i^{t-1}, message đã nhận, validation, share cũ và thông tin công khai; aggregate thành **model phục vụ** w_i^t.
3. Thị trường phản hồi dựa trên model phục vụ w^t; quan sát share riêng sigma_i^t, quyết toán payoff cho action a_i^{t-1} với cost đã ghi nhận.
4. Quan sát hiện tại o_i^t rồi chọn a_i^t. Tất cả node chọn trước khi tạo/nhìn message m^t của lượt này.
5. Local training: ell_i^t=L_i(w_i^t,D_i^{train};xi_i^t). Tạo m_i^t=ell_i^t nếu D, hoặc A_i(ell_i^t,o_i^t) nếu A. Giữ ell_i^t làm model local; gửi m_i^t cho lượt sau và ghi cost.

Không có thứ tự ưu tiên giữa các node. Nhìn message cũ rồi chọn message tiếp theo không phải nhìn action kín hiện tại của đối thủ. Model phục vụ w_i^t được cố định cho lần đánh giá thị trường t; baseline chưa phục vụ một model khác giữa lượt.

### 6.2. Bootstrap và kết thúc

Tại t=0, bắt đầu từ w_i^0 được công bố trong cấu hình, chưa có message đến; bước aggregation dùng model riêng. Môi trường tạo share ban đầu sigma_i^0, không coi đó là hậu quả của a_i^0. Node chọn action đầu tiên từ observation này, train và gửi.

Có T quyết định a^0,...,a^{T-1}. Tại t=T nhận và xử lý message cuối, aggregate/đánh giá thị trường và quyết toán u_i^T, **không chọn action mới**. Trạng thái t=T vẫn có W và market assessment; cost đó phải được phân bổ vào cost chuyển tiếp cuối theo cùng quy ước với các lượt trước.

Cost c_i^t được quy cho chuyển tiếp t->t+1: gồm phần tạo/gửi ở t và phần xử lý/phục vụ được tính phí ở t+1 nếu thí nghiệm tính chúng. Không tính hai lần. Hậu quả sau T vẫn bị cắt theo horizon hữu hạn; kiểm tra độ nhạy T, không tuyên bố đã tính hết ảnh hưởng lan tỏa.

Không dừng sớm vì mọi node tạm cùng chọn D. Gamma, T và lịch tau được công bố; T và các giá trị số chọn sau pilot. Khởi tạo cùng model giữa node trong baseline, các data split theo IID/non-IID, và cùng source/config/seed giữa đối chứng. Đề xuất p_i^0=0.5 và khảo sát initial conditions khác. Không gán identity cố định hoặc ép round đầu all-D.

### 6.3. Observation và memory

Policy pi_i(a|o_i;theta_i) có thể dùng: model local/model phục vụ đang có, local validation score, message vừa nhận, weight đã tính, share riêng hiện tại, t/tau và cost estimate được phép. W chỉ dùng tín hiệu có trước thời điểm nó thực thi như mục 5.

Node biết M và share riêng, không mặc định biết share từng đối thủ, quality thật, dữ liệu riêng, model giữ kín, action kín hoặc counterfactual payoff. Với hai node có outside option, share riêng không đủ suy ra riêng share đối thủ và outside share. Không mặc định công khai outside share cho mọi node.

Baseline không dùng history window, reputation, recurrent hidden state hoặc chuỗi message trong W/policy input. Tham số policy đang học và dữ liệu tạm để gán feedback trễ vẫn là trạng thái thích nghi; không thể vừa học theo kinh nghiệm vừa xóa toàn bộ trạng thái này. Simulator được giữ full state/log để tái lập nhưng không đưa state kín vào observation.

Extension reputation ở mục 13 là một **nhánh có memory được đặc tả riêng**, không sửa ngược kết quả baseline memoryless. State mở rộng thêm reputation theo quan hệ và lịch probe; node chỉ dùng row reputation do chính nó duy trì cùng trạng thái message/link local.

Không giả định observation hiện tại là Markov đủ. Global state phải gồm các model phục vụ/local, message đang chuyển, optimizer, cost đang chờ quyết toán, tau/time, dữ liệu và ngẫu nhiên cần thiết.

## 7. Payoff, best response và equilibrium

Utility function là quy tắc cố định; realized payoff là giá trị sau một chuyển tiếp; expected stage payoff lấy kỳ vọng theo ngẫu nhiên của chuyển tiếp đó. Continuation value còn tính các lượt sau.

Với hai node, p_i^{xy}(s;W)=E[bar u_i^{t+1}|s_t=s,a^t=(x,y),W].

| Node 1 / Node 2 | D | A |
| --- | --- | --- |
| D | (p_1^{DD},p_2^{DD}) | (p_1^{DA},p_2^{DA}) |
| A | (p_1^{AD},p_2^{AD}) | (p_1^{AA},p_2^{AA}) |

Các entry phụ thuộc state/market/cost/W, không là bốn hằng số cho toàn episode. Game một chuyển tiếp có thể liệt kê pure Nash, giải mixed bằng điều kiện indifferent. Best response dài hạn cần transition và policy đối thủ; full-state oracle chỉ dùng ngoại tuyến nếu vượt information boundary.

Đặt Delta Q_i(o;W)=Q_i(A|o,W)-Q_i(D|o,W): âm thì D tốt hơn, dương thì A tốt hơn, bằng 0 cần công bố tie-break. Trong một chuyển tiếp và thang chuẩn hóa:

\[
\Delta\bar u_i=\Delta\sigma_i-\Delta\bar c_i.
\]

Không cộng thêm reward damage. W tác động quality rồi market share; giảm weight không mặc nhiên giảm lợi ích A đơn điệu. Nếu m gần 0 và message còn tạo lợi thế thị phần, A có thể vẫn hấp dẫn. Phân biệt miền giảm damage với miền thật sự răn đe.

### Ví dụ số theo utility thị trường hiện tại

Một nhóm khách hàng, kappa=10, tau=0.70, cost chuẩn hóa D=0.01 và A=0.04. Giả lập quality sau một chuyển tiếp:

\[
q_i^{plain}=0.80-0.30\,\mathbf1[a_j=A],
\qquad q_i^{weighted}=0.78-0.01\,\mathbf1[a_j=A].
\]

Đây là hậu quả giả lập, **không** phải hiệu lực đã chứng minh của một W cụ thể. Defense minh họa có clean degradation 0.02 và giảm damage; cost W bổ sung trong ví dụ được giả định bằng 0, không được suy ra defense thật là miễn phí.

| Profile | Quality plain | Payoff plain | Quality weighted | Payoff weighted |
| --- | --- | --- | --- | --- |
| (D,D) | (0.80, 0.80) | (0.41232, 0.41232) | (0.78, 0.78) | (0.39828, 0.39828) |
| (A,D) | (0.80, 0.50) | (0.66538, 0.02512) | (0.78, 0.77) | (0.38478, 0.37436) |
| (D,A) | (0.50, 0.80) | (0.02512, 0.66538) | (0.77, 0.78) | (0.37436, 0.38478) |
| (A,A) | (0.50, 0.50) | (0.06651, 0.06651) | (0.77, 0.77) | (0.36055, 0.36055) |

Payoff tính bằng share có outside option trừ cost, làm tròn 5 chữ số; không dùng q_i-0.5*q_j. Trong game số này A strictly dominant với plain, D strictly dominant với weighted. Đây chỉ là Nash của game minh họa một bước, chưa phải equilibrium của DFL động.

## 8. Quality toàn mạng là mục tiêu thiết kế defense

Không đồng nhất mục tiêu kinh doanh của player với tiêu chí an toàn người nghiên cứu. Outside option và kỳ vọng ngoại sinh tạo động cơ cải thiện quality, nhưng không bảo đảm mọi nhóm/node đều được bảo vệ. Không cấp điểm test toàn mạng làm reward chung để ép kết quả.

### Các điều chỉnh bối cảnh đề xuất

- Giữ tổng khách hàng tiềm năng và nhóm khách hàng cố định, cho phép không sử dụng dịch vụ; do đó mọi model cùng kém có thể làm tổng nhu cầu giảm.
- Đánh giá quality tuyệt đối, không lấy thứ hạng/average quality làm kỳ vọng khách hàng.
- Dùng nhóm thị trường cố định và mọi node cùng đủ điều kiện phục vụ; chưa cho chọn bỏ nhóm khó hoặc rời episode để làm metric đẹp hơn.
- Dữ liệu đánh giá có đại diện các nhóm mục tiêu; final test không tham gia market simulator, W hoặc policy.
- Không thêm subsidy, fine hoặc global-quality bonus nếu chưa có cơ chế quan sát/chi trả và câu hỏi riêng.

Đây là căn cứ kinh tế cho quality, không phải thủ thuật sửa utility để có all-D. Nếu quality vẫn giảm thì phải giữ kết quả âm.

### Metric chính

Gọi q_ig^{test,t} là quality model phục vụ i tại t trên test nhóm g, chỉ dùng đánh giá. Chọn mu_g cố định trước; không dùng thị phần biến thiên làm trọng số:

\[
Q_{net}^t=\frac1n\sum_{i=1}^n\sum_g\mu_g q_{ig}^{test,t},
\qquad
Q_{min}^t=\min_i\sum_g\mu_g q_{ig}^{test,t}.
\]

Báo trajectory và mean quality theo thời gian 1..T, final Q_net, Q_min và kết quả theo nhóm/node. Đây là quality của tập model riêng cố định, không phải accuracy của một global model tưởng tượng. Nếu cần chất lượng theo phân phối cục bộ, báo thêm như metric khác, không đổi phân phối đánh giá theo cơ chế đang thắng.

Đối chiếu: clean DFL không defense, clean DFL có W, attacked DFL không defense, attacked DFL có W, và local-only. Có cả adaptive A/D lẫn attacker luôn-A; dùng cùng attack/cost/data/initial conditions khi so sánh phù hợp.

Chọn W dựa trên chất lượng dưới tấn công, đồng thời công bố giới hạn clean degradation, chi phí và mức giữ lợi ích học liên kết. Ngưỡng/metric chính chốt bằng validation và pilot, không chọn trên final test. Tỷ lệ A và best-response advantage là bằng chứng incentive đi kèm, không thay quality.

Nếu chọn backdoor phải thêm attack success rate và clean quality; bối cảnh quality thị trường hiện tại chưa mặc nhiên biểu diễn lợi ích của backdoor hiếm kích hoạt. Task/attack mục tiêu cần chốt trước claim đó.

## 9. Phần D — EGT, replicator dynamics và quy tắc thích nghi

### 9.1. Replicator dynamics cần những gì?

1. **Đối tượng tiến hóa:** chiến lược A/D tại một game đã cố định, hay policy qua nhiều lượt. Không đồng nhất tần suất action trong một trajectory với tần suất loại policy trong quần thể.
2. **Fitness có nghĩa:** expected net payoff/return của mỗi chiến lược với cùng điều kiện khởi đầu, W, market, cost và phân bố đối thủ. Không trộn payoff của task dễ với task khó mà coi đó là hiệu quả action.
3. **Quy tắc tương tác/cập nhật:** well-mixed hay graph, quần thể hữu hạn hay giới hạn liên tục, và tốc độ thích nghi.
4. **Cơ chế tạo dynamics:** reproduction/imitation có thông tin thực sự quan sát được, hoặc một quy tắc học cục bộ có giới hạn kỳ vọng liên hệ với replicator. Không cấp nhãn A/D và utility thật của neighbor cho node chỉ vì simulator biết.
5. **Điều kiện ổn định của môi trường:** đóng băng state/game để phân tích, hoặc chứng minh xấp xỉ tách thang thời gian. Tau đang tăng, cost thay đổi và model đang học không mặc định cho một payoff matrix dừng.

### 9.2. Tầng lý thuyết khởi đầu

Với game đối xứng well-mixed đồng nhất đã cố định, gọi x là tỷ lệ chiến lược A:

\[
\dot x=x(1-x)[f_A(x;W)-f_D(x;W)].
\]

Nếu payoff matrix hàng/cột là (A,D), thì f_A=x*p_AA+(1-x)*p_AD và f_D=x*p_DA+(1-x)*p_DD. Tìm nghiệm biên/nội tại, xét dấu dynamics và khả năng A xâm nhập khi x gần 0. All-D đứng yên ở biên không tự chứng minh ổn định; phải kiểm tra A hiếm có lợi không. Mutation dương thường thay absorption tuyệt đối bằng một phân bố dài hạn; phải định nghĩa kernel mutation nếu dùng.

Với graph/utility/data không đồng nhất, có thể phân tích từng node có mixed action p_i tại một state cố định:

\[
\dot p_i=p_i(1-p_i)\big[f_{iA}(p_{-i};s,W)-f_{iD}(p_{-i};s,W)\big].
\]

Đây là hệ coupled theo node, không rút đúng về một x toàn mạng nếu chưa có giả định đóng phù hợp. f_ia là kỳ vọng theo profile đối thủ và chuyển tiếp tại cùng s; game nhỏ có thể liệt kê chính xác. Với state cố định, nghiệm ổn định chỉ là kết luận cục bộ của game đó; không suy ra ESS hay hội tụ toàn hệ học.

### 9.3. Có thể liên hệ replicator với feedback riêng mà không nhìn đối thủ

**Baseline đã chọn cho pilot**, chưa thay learner dài hạn: xét game một bước cố định với bar u nằm trong [-C,1], C là upper bound cost chuẩn hóa công bố trước. Đặt R=(bar u+C)/(1+C) trong [0,1]. Cùng phép biến đổi áp dụng cho A và D; nó không thay thứ tự payoff. Pilot dùng C=0.006, eta=0.1 và p_i^0=0.5.

Node lấy a_i theo Bernoulli(p_i), nhận payoff của chính action đã chọn, rồi cập nhật:

\[
p_i'=p_i+\eta R_i\big(\mathbf1[a_i=A]-p_i\big),
\qquad 0<\eta\leq1.
\]

Cập nhật giữ p_i trong [0,1]. Tại cùng state/game và với profile đối thủ lấy mẫu theo p_-i:

\[
\mathbb E[p_i'-p_i\mid p,s]
=\frac{\eta}{1+C}p_i(1-p_i)(f_{iA}-f_{iD}).
\]

Vì vậy đây là một cách tạo **replicator-shaped expected drift** từ payoff cá nhân, không đòi quan sát counterfactual hoặc payoff đối thủ. Không cần thêm leaderboard/imitation để hợp thức hóa công thức. Phải chọn bound C hợp lệ; nếu cost không bị chặn thì không dùng phép chuẩn hóa này như đã viết, cũng không clip reward tùy tiện để che lỗi.

Giới hạn: rule trên so payoff một chuyển tiếp, không tự tối ưu J dài hạn. Payoff trễ phải gắn đúng action/p đã dùng; p=0 hoặc 1 là absorbing nếu không có exploration/mutation. Sampling noise, bước eta hữu hạn, thay đổi state và market làm trajectory khác ODE; không có bảo đảm Nash từ đẳng thức kỳ vọng. Nếu thêm exploration, công bố nó và kiểm tra cả dynamics thay vì gọi đó là replicator thuần.

### 9.4. Learner dài hạn và vị trí của history

Đề xuất đã bàn: policy không recurrent pi_i(a|o_i;theta_i), giữ tham số cố định trong episode; cuối episode cập nhật từ action và return-to-go riêng gồm cost và share về sau. Đây là reinforcement-based adaptation, không tự gọi là replicator/Fermi hoặc khẳng định tương đương 9.2. Cần định nghĩa policy class, thuật toán, estimator và credit assignment trước triển khai.

Nếu muốn dùng replicator trực tiếp trên **policy dài hạn**, phải định nghĩa một tập policy, cách tính expected episode return cho từng policy trong cùng phân bố đối thủ, quy tắc thay đổi tần suất và ai có được các thông tin đó. Tính payoff các policy chưa chạy bằng oracle chỉ hợp lệ cho tầng phân tích; không lén đưa nó vào agent cục bộ.

Không cần đổi bối cảnh thành công khai accuracy/utility/action thật chỉ để sử dụng EGT. Baseline không có message history vẫn được giữ; learning parameter và bộ nhớ tạm để quyết toán action/return là cần thiết. Mở rộng reputation/history là hướng sau, không phải đã triển khai.

**Quyết định D sau pilot/game nhỏ:** giữ rule một bước 9.3 làm baseline runtime có liên hệ expected drift với replicator; action chỉ lấy mẫu từ p_i, cập nhật bằng payoff riêng, chưa dùng feature model/message để chọn role. W vẫn dùng local observation. Đã tính oracle best response 1–3 bước và thấy quyết định khác myopic; learner dài hạn local 9.4 vẫn cần thiết kế. Không coi baseline một bước là solver của J hoặc biến oracle thành policy triển khai.

## 10. Cấu hình cụ thể sau pilot và việc phải làm tiếp

### Task, dữ liệu và môi trường

MNIST phân loại 10 chữ số **0–9**, model MLP 784→64 ReLU→10 logits, SGD lr=0.1, batch=64, 1 local epoch/round. Pilot 4 node ring, 1,024 train và 200 validation samples/node, initialization chung.

Chỉ dùng official train 60,000 ảnh; stratified pool khoảng 80% train, phần còn lại chia ba pool local-validation/market/pilot-audit. IID chia từng label gần đều; non-IID dùng class-to-client Dirichlet alpha=1.0 (alpha=0.5 chỉ dùng cho artifact 4-node cũ) và dùng cùng proportions cho train/validation, nhưng index rời nhau. Với 16 node, pool validation mặc định 4,005 ảnh không đủ 200/node, nên layout `balanced` cấp market/pilot-audit đúng `eval_per_class` mỗi lớp và chuyển phần dư (~6,000 ảnh) sang local-validation (10,005 ảnh); layout `legacy` giữ nguyên cho artifact cũ. Số sample mỗi node được giữ cố định. Không resample split âm thầm nếu không đủ dữ liệu.

Market và pilot-audit mỗi tập 1,000 ảnh disjoint, 100/lớp; 10 nhóm chữ số với mu_g=0.1. Official test 10,000 ảnh chưa download/sử dụng. Pilot-audit là dữ liệu development, không được gọi là final-test. Market chỉ tạo feedback share, audit chỉ đo quality. Index, label counts, source checksum và log đầy đủ nằm trong JSON pilot.

M=1,000 khách hàng tiềm năng là quy ước kinh tế; dùng expected share, không lấy mẫu số khách hàng. Kappa=10; tau_t=min(0.8,0.1+0.05t). Các giá trị này cố định trong một run và còn cần sensitivity.

### Operator A và candidate W0

\[
m_i^t=
\begin{cases}
\ell_i^t, & a_i^t=D,\\
w_i^t-5(\ell_i^t-w_i^t), & a_i^t=A.
\end{cases}
\]

Đây là sign-flipped local update với amplification cố định, không phải label flipping/backdoor hoặc attacker tự tối ưu mọi tham số. Model local ell_i^t không bị thay bằng bản message độc.

Tại receiver, gọi L_i là cross-entropy trên validation riêng. Tập message được nhận:

\[
\mathcal G_i^t=
\{j\in\mathcal N_i:
L_i(m_j^{t-1})\leq L_i(\ell_i^{t-1})+1.0\}.
\]

W0 đặt self-weight=1/(1+|G_i^t|), weight mỗi neighbor được nhận bằng cùng giá trị đó, còn lại bằng 0. Không dùng nhãn A/D thật để quyết định. Rule giống nhau cho A và D; self-only là fallback hợp lệ nhưng không được coi mặc nhiên là defense thành công.

Dung sai 0.1/0.5 làm smoke non-IID mất lợi ích học chung, nên chọn 1.0 từ development calibration; chưa chứng minh đó là optimum hoặc robust với attack khác. W0 chỉ là reference candidate. Contribution chiến lược phải đến từ phân tích incentive và thiết kế/sửa rule tiếp theo, không đổi tên reference gate thành một phương pháp mới.

Cost chuẩn hóa pilot: c_D=0.005 và c_A=0.006, bound C=0.006. Đây là giả định kinh tế, không phải cost runtime đo được. Game nhỏ đã so cost-gap=0/0.001/0.01/0.05; cost cao chỉ là sensitivity, không chọn để ép D. Pilot T=20 và settlement cuối; game oracle đã dùng gamma=0.95, horizon 1–3. Learner pilot một bước vẫn không tối ưu discounted long-horizon return.

### Evidence và bước tiếp theo

Code: [mnist_pilot.py](../experiments/mnist_pilot.py); cấu hình/kết quả đầy đủ: [ROADMAP.md](ROADMAP.md), [mnist_pilot.json](../experiments/results/mnist_pilot.json). Pilot cho clean learning có ích, attack gây damage và W0 phục hồi phần lớn quality. Tỷ lệ A vẫn quanh 0.5: **chưa có bằng chứng răn đe hoặc equilibrium**.

Đã dựng game hai node từ snapshot chung, liệt kê bốn action profile, giải stage pure/mixed Nash và oracle best response 1–3 bước, phân tích replicator theo node trong game đóng băng. Kết quả reference tại mục 11; candidate W1 và giới hạn incentive tại mục 12. W0/W1 chưa răn đe được poisoning trong các state vẫn chia sẻ hữu ích.

Phải giữ kiểm tra cost-gap/tau/initial-condition sensitivity, attack luôn-A, clean W0 và local-only; báo Q_net/Q_min, từng nhóm/node, utility, outside share, switching và overhead. Khóa protocol trước final-test; không dùng test để tạo thị trường hoặc chọn W.

Không ép all-D, switching hoặc market recovery. Kết quả âm và miền defense chỉ giảm damage mà chưa răn đe là kết quả hợp lệ. Chưa mở rộng hệ thống lớn trước khi game nhỏ và incentive được kiểm chứng.

## 11. Game đo từ MNIST — scope và kết quả để thiết kế defense tiếp

Chương trình [mnist_game.py](../experiments/mnist_game.py), [raw game data](../experiments/results/mnist_game.json), [verification](../experiments/results/mnist_game_verification.json). Dùng native split hai node; snapshot sau 0/5/10 round all-D uniform cho seed 7/19/31 và IID/non-IID. Mọi cơ chế bắt đầu từ cùng model snapshot có SHA-256, không warmup riêng rồi so các state khác nhau.

Mỗi snapshot sinh cây 4 nhánh/decision node, depth=3: 85 state gồm root, 84 transition. Tổng 18 snapshot × uniform/W0/self-only = 54 cây. Training dùng cùng seed/node/absolute-round giữa các nhánh. Giữ model snapshots, data indices, model hashes, per-group market/audit quality và weight từng transition.

### Scope của solver và replay

- Oracle biết state và các seeded transition được liệt kê; kết quả exact cho **finite deterministic seeded game** đó, không phải expectation toàn bộ ngẫu nhiên SGD hoặc policy dùng observation cục bộ.
- Đối thủ fixed-policy: always-D, always-A, alternating D/A, iid_half. Oracle chọn trước action hiện tại của đối thủ; iid_half được lấy kỳ vọng, không được nhìn coin trước khi chọn.
- Giải pure/mixed Nash cho stage 2×2; lưu equilibrium cells để giữ continuum ở payoff bằng nhau. Best response nhiều bước không được gọi là dynamic Nash.
- Replay cost-gap và tau trên cùng model tree hợp lệ vì uniform/W0/self hiện không dùng share/cost để quyết định weight. Nếu W tương lai dùng chúng, phải tái dựng transition theo state đó, không reuse tree như hiện tại.
- Replicator dùng hai mixed probabilities theo node, không symmetrize utility/data khác nhau. Jacobian/eigenvalues chỉ kết luận trong frozen game; zero real part/nonhyperbolic chưa đủ để kết luận stability. Không phân loại đầy đủ stationary continua.

### Kết quả và điều kiện thiết kế tiếp

Ở cost-gap=0.001, tau scheduled, uniform cho AA tại 18/18 snapshot. W0 cho AA ở 12, AD/DA ở 3, DD ở 3. Ba DD đều là cold-start non-IID mà W0 từ chối mọi message D/A và trùng self-only; không coi đó là răn đe vẫn giữ được lợi ích học chung. Self-only với cost-gap=0 làm mọi mixture indifferent, không phải strictly stable all-D.

Ở horizon 3, trong 144 bài toán response/cơ chế, uniform có 38 root myopic A→oracle D; W0 có 5 A→D và 10 D→A. Hậu quả dài hạn có thể đảo động cơ theo cả hai hướng. Các bài toán có tương quan theo snapshot và policy; đây không phải tần suất quần thể hay kết luận thống kê tổng quát.

Kiểm độc lập khớp 432 best responses horizon 3 bằng liệt kê 128 contingent policies; lịch sử action đối thủ trong miền kiểm chứng là thông tin oracle, không cấp cho runtime local. Kiểm expected regret của 802 equilibrium cells, snapshot hashes và 5 regression tests được ghi ở roadmap/verification.

Đã thử W1 theo hướng này, xem mục 12. Giữ W0 làm baseline; không tăng cost, đổi reward hoặc cho self-only thành mặc định chỉ để có DD.

## 12. W1: chọn mixture bằng validation local và giới hạn incentive

### Rule và information boundary

Candidate `validation_mix` chỉ triển khai cho game **hai node**, chưa tích hợp vào pilot ring bốn node. Với j là neighbor duy nhất của i:

\[
\Lambda=\{0,\ 0.125,\ 0.25,\ 0.5\},\qquad
\lambda_i^*=\operatorname*{argmin}_{\lambda\in\Lambda}
L_i\big((1-\lambda)\ell_i+\lambda m_j\big).
\]

Chọn lambda nhỏ nhất khi loss bằng nhau; weight self=1−lambda, neighbor=lambda. Grid được chốt trước run, không sweep trên audit để chọn điểm đẹp. So với W0 đánh giá cả message rồi nhận với weight 0.5, W1 đánh giá **model thực sự được phục vụ sau aggregate** và có thể chọn một phần message hữu ích. Không có tolerance dương hoặc lower bound ép chia sẻ.

Rule chỉ dùng model local giữ sạch, message nhận và validation local. Không đọc action thật, quality market/audit, share, cost, model riêng của đối thủ hoặc history. Cả A/D dùng cùng rule. Cây model vẫn replay được theo cost/tau vì W1 không phụ thuộc hai biến đó.

Do grid chứa 0, loss validation của mixture được chọn không lớn hơn loss model local giữ lại **trên chính tập validation dùng chọn weight**. Đây không phải bảo đảm accuracy trên market/audit, generalization, giữ lợi ích non-IID hay incentive. Việc dùng lặp lại validation để chọn weight cũng chưa được chứng minh không overfit.

W1 cần bốn loss evaluations/receiver/transition trên grid này; W0 hai node cần loss self và loss message. Cost utility vẫn cố định như đối chứng để cô lập hiệu ứng rule; chưa quy overhead tính toán thành chi phí kinh tế.

### Vì sao loại message A vẫn chưa loại động cơ A?

Trong một transition hai node với training seed cố định, action riêng chỉ đổi bản gửi. Trước action đối thủ cố định, model phục vụ tiếp theo của i không đổi khi i chọn A thay D; receiver j có thể đổi model. Đặt q_ig là quality market của i sau transition, và q_jg^D/q_jg^A là quality của j theo bản gửi của i:

\[
b_{ig}=e^{\kappa(q_{ig}-\tau)},\qquad
b_{jg}^{x}=e^{\kappa(q_{jg}^{x}-\tau)}.
\]

\[
\Delta u_i
=\sum_g\mu_g b_{ig}
\left[
\frac{1}{1+b_{ig}+b_{jg}^{A}}
-\frac{1}{1+b_{ig}+b_{jg}^{D}}
\right]-m_i.
\]

Nếu D giúp model đối thủ tốt hơn A ở mọi nhóm và tốt hơn nghiêm ở ít nhất một nhóm, phần lợi ích thị trường của A dương. Với m_i=0 thì A có lợi nghiêm. Khi A bị weight=0 nhưng D được nhận, A có thể có tác dụng **tước lợi ích hợp tác**, dù không làm receiver tệ hơn bản local giữ lại. Không thêm action “không gửi”; đây là hậu quả của action A đã có trong game.

Điều kiện theo từng nhóm ở trên quan trọng: chỉ biết average accuracy hoặc validation loss không đủ suy dấu Delta u. Đây là giới hạn có điều kiện của **stage game trong protocol hiện tại**, không phải định lý bất khả thi cho mọi defense, mọi horizon hoặc cơ chế có reciprocity/history.

### Kết quả development và quyết định

Run [mnist_w1_game.json](../experiments/results/mnist_w1_game.json) có 72 cây: 18 snapshot × uniform/W0/self/W1, cùng warmup và seed. [Verification và bảng tổng hợp](../experiments/results/mnist_w1_verification.json); kết quả chi tiết tại [roadmap §3.8](ROADMAP.md#38-w1--validation-trên-mixture-và-kết-quả-âm).

W1 loại cả 36 message A trong các unilateral root deviations, nhưng A vẫn có lợi ở 16/36 node-state. Sáu root DD đều self-only ngay cả khi nhận D. W1 giữ clean quality IID nhưng giảm clean quality non-IID so với W0; chưa đạt mục tiêu quality toàn mạng cộng incentive. Ở các state sâu hơn vẫn có message A được nhận, nên không tuyên bố chặn mọi poison.

Giữ W1 làm candidate/kết quả âm có thể tái lập, không chọn làm final defense. Extension reciprocity đã được dựng và kiểm ở mục 13; baseline memoryless vẫn giữ nguyên. Learner dài hạn cục bộ vẫn chưa được chốt.

## 13. Extension reputation–reciprocity: quyền nhận có điều kiện

### 13.1. Reputation phải gắn với tài nguyên nào?

Reputation chỉ giảm weight **của message người bị chấm thấp** vẫn là bảo vệ receiver; nó chưa tước update tốt mà người đó nhận. Extension dùng reputation để sender quyết định **có gửi cho recipient đó trong lượt sau hay không**.

Đây là thay đổi có điều kiện của protocol, không phải sửa utility: A/D vẫn chỉ quyết định parameter manipulation; các node tuân cùng rule chấm điểm/gửi/aggregate, kể cả khi chọn A. Không bổ sung một action chiến lược “tự khai reputation” hoặc “tùy ý bỏ qua hình phạt”.

Giả định bổ sung:

- Stable authenticated identities, không đổi identity/reset reputation tùy ý.
- Kênh song phương có thể kiểm soát quyền nhận; model/update không công khai, không có relay/collusion lấy lại update bị từ chối. Mạng hai node hiện tại không xét bypass qua node thứ ba.
- Reputation là đánh giá **trực tiếp, local, theo quan hệ**, không có server/global leaderboard hay chia sẻ lời tố cáo. Không cần tin reputation report của đối thủ.
- Protocol được biết; node có thể kỳ vọng hậu quả qua feedback/link local, nhưng không được cấp điểm reputation kín mà neighbor chấm cho mình hoặc quality/counterfactual của neighbor. Full-state oracle vẫn chỉ là công cụ nghiên cứu.

Nếu update vẫn public-broadcast cho mọi node, reputation không thể thực thi mất quyền nhận bằng cách chỉ đổi weight. Sybil, collusion, né delivery rule hoặc tự đổi receive-side defense nằm ngoài bảo đảm của extension này.

### 13.2. Điểm đóng góp và reputation

Gọi R_ij^t∈[0,1] là điểm **node i chấm node j**, trước action t. Ban đầu R_ij=1; diagonal không dùng.

Khi i nhận message j, dùng cùng grid Lambda ở mục 12:

\[
d_{ij}^t=L_i(\ell_i^t)
-\min_{\lambda\in\Lambda}L_i((1-\lambda)\ell_i^t+\lambda m_j^t),
\qquad z_{ij}^t=\mathbf1[d_{ij}^t>10^{-6}].
\]

\[
R_{ij}^{t+1}=(1-\beta)R_{ij}^{t}+\beta z_{ij}^t,\qquad \beta=0.5.
\]

Không có message thì giữ R_ij, **không gán score 0**: vắng message có thể do chính rule withholding, không phải hành vi xấu của peer. Điểm chỉ phản ánh usefulness trên validation của evaluator; không đồng nghĩa xác suất peer là attacker. D hợp lệ nhưng không cải thiện validation vẫn bị score 0, còn A có ích trên tập này vẫn có thể được score 1.

### 13.3. Quyền nhận, timing và phục hồi

Sender i gửi cho recipient j tại action round t khi:

\[
P_{i\to j}^t=\mathbf1[R_{ij}^t\geq0.75\ \lor\ t\bmod2=0].
\]

Đây là **row sender/evaluator, column recipient**: i chấm j kém → i không gửi update cho j, không chỉ từ chối message j. Probe công khai mỗi hai round mở lại kênh để có bằng chứng phục hồi; không blacklist vĩnh viễn hoặc miễn phí reset identity. Parameter beta/threshold/probe được chốt trước run, chưa chứng minh tối ưu.

Thứ tự một transition:

1. Từ reputation có sẵn, xác định permissions; tất cả action A/D hiện tại vẫn kín.
2. Local train, tạo bản D/A và gửi chỉ trên hướng được phép.
3. Nhận message, chấm usefulness và cập nhật reputation **cho lượt kế tiếp**.
4. Aggregate bằng **W0** trên các message thật sự nhận, rồi market settlement như cũ. Không có message thì giữ model local sạch.

W1 grid chỉ dùng làm estimator đóng góp; receiver vẫn dùng W0 để so riêng hiệu ứng access với reference có clean non-IID quality tốt hơn. Việc chấm ở bước 3 không được thu hồi message đã gửi trong bước 2. Không đổi quyền nhận hiện tại dựa trên current action của đối thủ.

Simulator ghi full matrix để phân tích; runtime node chỉ sở hữu row của mình và tình trạng kênh local. State global thêm R và probe phase; không còn là baseline không history. Snapshot model sau warmup 0/5/10 được giữ nguyên, nhưng R được reset về 1 ở mỗi root: đây là thí nghiệm khởi tạo reputation lạc quan, không giả vờ tái dựng history đã học.

Cost vẫn c_D=0.005/c_A=0.006 dù có message bị withholding, nhằm cô lập tác dụng protocol; chưa đo/calibrate chi phí compute/communication thật. Rule chấm điểm cần thêm các validation evaluations, không được gọi là miễn overhead.

### 13.4. Điều kiện incentive, không giả định quality tự giảm

Với cùng opponent policy, so hai trajectory: D hiện tại rồi theo cùng continuation policy, hoặc A hiện tại rồi theo continuation đó. Đặt G_i là gain payoff ở settlement đầu, và:

\[
L_i=\sum_{k=1}^{H-1}\gamma^{k-1}
\mathbb E[u_i^{D,k+1}-u_i^{A,k+1}],
\qquad J_i^A-J_i^D=G_i-\gamma L_i.
\]

A không có lợi trong phép so này khi gamma*L_i≥G_i. L_i là **mất utility tương lai ròng**, đã gồm thay đổi quality/share của chính mình và đối thủ, cost và reputation/delivery trong hai trajectory. Không đồng nhất L_i với số lượt bị chặn hoặc tự giả định L_i dương.

Cần kiểm riêng ba mắt xích:

- Update bị mất phải có ích cho model người mất. Bị cô lập không có nghĩa accuracy tuyệt đối giảm: local SGD vẫn có thể cải thiện; effect cần so với trajectory có quyền nhận.
- Quality riêng thấp hơn chỉ đảm bảo share thấp hơn khi quality đối thủ và điều kiện thị trường giữ nguyên. Trong hai trajectory thật, đối thủ cũng đổi quality nên phải đo share, không suy từ Q_i một mình.
- Mất share/utility tương lai phải đủ lớn và đủ sớm để bù gain hiện tại; finite horizon có lượt cuối không còn sanction tương lai. Không suy all-D mọi round hoặc equilibrium từ một ví dụ phạt được.

Nhánh đối chứng `reputation_only` chấm cùng score nhưng không chặn quyền nhận. So cùng action sequence giữa hai nhánh tách effect của enforcement khỏi tự gây damage/đổi training do action A. So A trajectory với all-D **trong từng mechanism** mới đo lợi thế tương đối, không dùng utility attacker giảm so baseline làm bằng chứng đủ.

### 13.5. Kết quả và vị trí nghiên cứu

Code [mnist_reciprocity.py](../experiments/mnist_reciprocity.py), [raw data](../experiments/results/mnist_reciprocity.json), [verification](../experiments/results/mnist_reciprocity_verification.json). Chạy 18 root × monitor/reciprocal, exact oracle H1–3 và 5 fixed-action policies × 20 transition. Chi tiết ở roadmap §3.10.

Enforcement thực sự tạo withholding trễ; monitor khớp W0. Nhưng all-D non-IID vẫn bị chặn 133/360 directed deliveries; mean clean Q_net giảm khoảng 2.09 điểm phần trăm. Một lần A bớt lợi ở một số state, còn always-A vẫn có lợi ở 9/9 IID và 7/9 non-IID snapshot trước opponent always-D trong phép so fixed policies 20 bước. Chưa có cơ chế răn đe đạt yêu cầu.

Ý tưởng reputation phân bổ update đã có tiền lệ: [CFFL — Lyu, Xu, Wang, 2020](https://arxiv.org/abs/2008.12161), đặc biệt §3.2 dùng server reputation để phân bổ aggregated updates ở round sau; §5 nêu tradeoff threshold và loại nhầm participant. Bản này dùng direct bilateral evaluation thay server, nhưng **“thêm reputation” không tự là novelty** hoặc bằng chứng chống strategic poisoning trong thị trường cạnh tranh.

Đã triển khai nhánh validation đại diện service, đối chứng với validation lệch cũ; protocol dữ liệu và boundary official test ở mục 14. Đây là assumption dữ liệu mới công bố riêng, không lén thay split gốc.

Sau đó cần đo cooperation surplus và sanction loss, probe/threshold sensitivity, reputation persistence và endgame trước khi chọn learner local dài hạn. Frozen stage replicator tại root chưa thấy hình phạt trễ; EGT có memory cần payoff theo trajectory/policy và state reputation, không tái dùng một payoff matrix tức thời làm chứng cứ ổn định.

## 14. Validation đại diện service và vai trò official test

### 14.1. Quyết định về official test

**Dùng official MNIST test 10,000 ảnh cho đánh giá cuối sau khi khóa protocol, chưa dùng trong development hiện tại.** Không dùng nó làm validation chấm reputation, chọn weight, tạo market share/payoff, chọn threshold/probe/attack hoặc chọn candidate theo kết quả.

Chỉ “node không nhìn test trực tiếp” là chưa đủ: nếu simulator tạo reward từ test để agent học, hoặc người nghiên cứu sửa cơ chế theo test score, test đã tham gia development. Download không tự tạo leakage; vấn đề là dữ liệu/kết quả được dùng vào quyết định nào. Run hiện tại vẫn chỉ load official train.

Nếu chủ động đánh giá official test trước khi chốt cơ chế, phải báo exploratory và không tiếp tục gọi nó là final holdout độc lập sau các lần sửa dựa trên kết quả. Khi đó cần một holdout chưa được dùng để thiết kế/chọn mô hình, không đổi nhãn pilot-audit thành final-test.

Trước final evaluation phải khóa candidate, data/seed protocol, parameter/cost/attack settings, horizons, policies, metrics và cách tổng hợp. Dùng đủ 10,000 test samples, báo cả micro accuracy và per-class accuracy; Q_net vẫn là mean qua node và mean qua 10 nhóm chữ số, không nhầm với micro accuracy khi số test samples mỗi lớp khác nhau.

### 14.2. Đối chứng validation với cùng budget

Hai nhánh `--validation-modes train_matched service`:

- `train_matched`: giữ nguyên 200 validation samples/node theo phân phối local của split cũ.
- `service`: mỗi node có 200 sample validation riêng, **20/lớp 0–9**, đại diện tệp service uniform đã định nghĩa trước. Đây là giả định có dữ liệu gán nhãn đánh giá service dù train lịch sử non-IID; không đồng nghĩa node được biết quality thật của đối thủ.

Service samples lấy từ **reserved local-validation population của official train 60,000**, trước allocation IID/Dirichlet, loại các validation samples cũ đã chọn rồi lấy không hoàn lại, disjoint giữa node. Không lấy từ train/market/audit populations và không dùng official test. Tập train 1,024/node, market, pilot-audit, initial model/warmup và action RNG giữ nguyên trong các đối chứng.

Không tăng validation sample budget hoặc chọn class weights theo audit; uniform 10 nhóm là service distribution đã có trong context. Giả định thu thập validation đại diện service chưa được tính thành economic cost mới; phải nêu nếu áp dụng thực tế.

Cả reputation score và W0 gate đều dùng validation đã chọn của nhánh. Vì vậy đổi nhánh có thể đổi cả filtering lẫn reputation, không được quy toàn bộ cải thiện cho reciprocity. Hai cơ chế monitor/conditional-access được giữ trong **mỗi nhánh validation** để tách tác dụng enforcement.

Code dùng chung `partition_pools` trong pilot, giữ nguyên thứ tự tiêu thụ RNG của `make_split`; `service_validation_split` chỉ thay validation. Manifests lưu actual indices/counts cùng source validation cũ để kiểm disjointness. Artifacts lịch sử giữ checksum source tại thời điểm run; refactor không được sửa ngược evidence cũ.

### 14.3. Kết quả đối chứng và quyết định sau run

Run [mnist_service_validation.json](../experiments/results/mnist_service_validation.json) có 18 snapshot × 2 validation modes × 2 access mechanisms = 72 cây H1–3, cùng 360 rollout 20 bước. [Verification](../experiments/results/mnist_service_validation_verification.json) lưu split provenance, paired quality/share/utility, contribution-credit alignment và phạm vi kiểm độc lập. Không thay cost, attack, beta, threshold, probe hoặc chọn lại grid theo kết quả.

Ở clean non-IID, service validation cho 360/360 message D được giao score useful, không chặn directed delivery nào, so với 133/360 bị chặn trong train-matched. Mean final audit Q_net của nhánh reciprocal tăng 75.10%→87.13%. Tuy nhiên monitor cũng tăng 77.19%→87.13%: phần lớn phục hồi đến từ validation ảnh hưởng W0 filtering; không gán toàn bộ cho reputation.

Trong service validation, một lần A rồi D trước always-D có mean gain J≈0.00572 ở non-IID, giảm so monitor≈0.08165 nhưng vẫn dương ở 5/9 snapshot. Always-A vẫn có gain dương ở 9/9 IID và 8/9 non-IID khi có reciprocity; chưa đạt răn đe. Đây là fixed-policy comparison 20 bước, không phải best response tối ưu 20 bước hoặc equilibrium.

Phát hiện thiết kế tiếp: score chấm **best hypothetical mixture** trong grid, còn W0 thực sự dùng weight 0 hoặc 0.5. Trong service/non-IID/node0-always-A, 70/180 message A được giao nhận credit useful, nhưng 30 trong 70 không cải thiện validation của aggregate thực tế dùng W0. Không quy mọi credit A là lỗi: 40 message còn lại thực sự cải thiện criterion validation, và action label A không được làm oracle để phạt.

Đã triển khai hai đối chứng align credit với aggregate thực tế tại mục 15. Official test vẫn không dùng để chọn cách sửa; cần tách damage/incentive và kiểm access/recovery/endgame trước final evaluation.

## 15. Credit theo aggregate thực tế: hai đối chứng tách biệt

### 15.1. Rule được giữ cố định trước run

| Rule CLI | Model được phục vụ khi nhận message | Criterion để credit usefulness |
| --- | --- | --- |
| `legacy_gate` | W0: self hoặc weight neighbor=0.5 | Best hypothetical mixture trong grid W1; giữ làm reference của mục 13–14 |
| `realized_gate` | W0 không đổi | Loss improvement của chính aggregate W0 thực sự dùng |
| `realized_mix` | W1 chọn mixture trên validation như mục 12 | Loss improvement của chính mixture được chọn và phục vụ |

Với một neighbor j trong game hai node, đặt A_i là **model sau aggregation thực tế**, không phải action A. Credit của hai rule mới:

\[
d_{ij}^{\mathrm{real}}
=L_i(\ell_i)-L_i(A_i),\qquad
z_{ij}^{\mathrm{real}}
=\mathbf1[d_{ij}^{\mathrm{real}}>10^{-6}].
\]

Nếu không có message thì giữ reputation như cũ. Nếu có message nhưng weight=0, aggregate bằng model local nên d_real=0; không credit cho một mixture không được dùng. EMA, permissions, probe và timing giữ nguyên mục 13. Không dùng nhãn A/D, quality market/audit hoặc payoff counterfactual để chấm.

`realized_gate` cô lập tác dụng sửa credit: có thể vẫn nhận mixture làm validation xấu đi do tolerance W0; không credit không có nghĩa đã ngăn damage đó. `realized_mix` đổi aggregation để loss validation được chọn không hơn self trên tập dùng chọn weight, nhưng không bảo đảm market accuracy, generalization, non-IID sharing hoặc incentive.

### 15.2. Miền thí nghiệm và observation

Hai rule mới chạy cùng 18 root, hai validation assumptions và monitor/conditional-access; legacy artifact ở mục 14 là reference, có smoke replay riêng để kiểm semantics không đổi. Cùng action sequences/seed, train/market/audit, cost, threshold, probe, horizon và reputation initialization. Không tune lại các tham số nếu kết quả chưa răn đe.

Đây vẫn là direct reputation trên kênh bilateral, A chỉ thao túng bản gửi và tuân protocol còn lại. Score ghi actual loss/weight/credit để kiểm từ model thật; không cấp những counterfactual logs của simulator cho local policy. Với nhiều neighbor, marginal credit assignment khác vấn đề một neighbor này; chưa mở rộng rule hoặc claim sang topology đó.

CLI `--rules` chọn reference/candidate, default vẫn là `legacy_gate` để chạy reference có chủ đích, không phải chọn final defense. Artifact mới ghi rule từng game/rollout và source checksum; không ghi đè evidence lịch sử.

### 15.3. Kết quả và giới hạn sau khi align credit

[Raw run](../experiments/results/mnist_realized_credit.json), [verification/tổng hợp](../experiments/results/mnist_realized_credit_verification.json); kết quả chi tiết ở roadmap §3.13. Hai rule mới × 18 root × 2 validation modes × monitor/reciprocal = **144 cây H1–3 và 720 rollout 20 bước**. Legacy reference được giữ nguyên và replay riêng 8 cây/40 rollout.

46,582 delivered receiver cases trong tree/rollout có credit khớp loss improvement của model được phục vụ. Không còn credit theo mixture không được dùng ở hai rule mới; 24 real-model transitions được retrain và kiểm loss/argmin trực tiếp. Đây là consistency của criterion validation, không phải chứng minh credit là quality thật hoặc A không còn có lợi.

Với service validation, `realized_mix` + reciprocity đạt mean clean audit Q_net khoảng 89.31% IID/87.54% non-IID; clean non-IID có 0/360 directed deliveries bị chặn và mọi delivered D đều đem lại positive realized validation improvement. Không phải self-only solution. `realized_gate` + service vẫn chặn 26/360 clean non-IID deliveries, mean Q_net 85.60%: chấm credit đúng không tự sửa việc W0 nhận aggregate làm tăng loss.

Persistent node-0 always-A trước node-1 always-D vẫn có gain J so all-D ở 7/9 IID và 7/9 non-IID service snapshots với `realized_mix`. H3 oracle xét **cả hai player**, trước always-D, vẫn strictly chọn A tại root ở 9/18 IID và 12/18 non-IID player-state. Không có service snapshot nào mà nguyên sequence all-D không có profitable H3 deviation ở cả hai player.

Không lấy kết quả train-matched/non-IID/node0 có zero profitable fixed-policy A làm success claim: monitor đã có kết quả đó, clean Q_net thấp, và H3 vẫn có profitable root A ở player-state khác. Fixed node-0 stress tests không đại diện mọi node hoặc mọi policy.

### 15.4. Giới hạn access tiếp theo: probe vẫn phát update tốt

Probe công khai mỗi 2 round gửi message bất kể reputation. Trong 20 transition, mỗi node nhận tối thiểu 10 message từ peer qua probe, kể cả luôn đóng góp bị chấm thấp. Với `realized_mix` + service, node 0 always-A vẫn nhận **93/180 message D** qua 9 snapshot/mode; 90 là probe vô điều kiện. Đây là receipt/opportunity, không mặc định mọi message được aggregate hoặc hữu ích.

Bound này do protocol, không phụ thuộc solver; nó cho thấy cơ chế hiện tại chỉ giảm access chứ không cô lập. Nó **không tự chứng minh** probe là nguyên nhân duy nhất của gain A; hậu quả của model đã nhận trước đó, opponent quality, horizon và cost cũng tham gia J.

Đã chạy access/probe–recovery sensitivity tại mục 16, không đổi utility/cost hoặc bỏ recovery. Candidate chưa được chọn làm final defense; endgame và learner/equilibrium claim vẫn giữ giới hạn đã công bố.

## 16. Access/probe sweep với credit và aggregation cố định

### 16.1. Protocol được chốt cho sensitivity

Giữ `realized_mix`, private service validation, beta=0.5, threshold=0.75, epsilon=1e-6, cùng cost/attack/gamma và root models. Chỉ thay probe period K∈{1,2,4,8}; **không thay utility/cost hoặc bỏ recovery để ép D**. Nhánh train-matched không chạy lại trong sweep này; failure dưới assumption đó vẫn giữ ở evidence trước.

Một monitor enforcement-off được chạy cho mỗi root; reciprocal chạy riêng từng K. Monitor ghi K=1 nhưng không dùng reputation để chặn, nên không cần lặp baseline đó theo K. K=1 reciprocal phải khớp monitor: mọi action round đều là probe.

Probe dùng **absolute round** như protocol cũ, không reset phase về 0 tại snapshot. Các snapshot sau 0/5/10 warmup round giữ nguyên model, data và optimistic reputation=1. Do đó phase theo K cũng khác giữa roots; phải báo cùng snapshot chứ không chọn lại phase đẹp.

Số probe trong T transition từ absolute round s:

\[
N_{\mathrm{probe}}(s,T,K)
=\left|\{d\in\{0,\ldots,T-1\}:(s+d)\bmod K=0\}\right|.
\]

Mỗi node nhận ít nhất số message này từ peer trong reliable bilateral protocol, bất kể reputation. Receipt không đồng nghĩa positive aggregation weight hoặc quality gain. Tăng K giảm sàn access nhưng có thể làm evidence/recovery chậm hơn; đây là tradeoff phải đo, không phải bảo đảm răn đe.

### 16.2. Policies hai phía và endgame

Mỗi condition chạy H1–3 oracle như trước và 20-transition fixed schedules:

- All-D và iid_half.
- Một lần A rồi D, always-A, ba lần A rồi D, và chỉ A ở lượt cuối; mỗi kiểu đều chạy riêng actor=node 0 và actor=node 1, peer luôn D.

Đây là stress policies có chủ đích, không gán identity A/D bất biến cho game hoặc gọi chúng là learned equilibrium. Burst rồi D kiểm consequence/recovery của switching; D hợp lệ không được mặc định luôn useful.

Mọi schedule giống nhau giữa các K và monitor, cùng training/action RNG. So J_actor(deviation) với J_actor(all-D) **trong cùng condition**, rồi mới so các K; không lấy attacker utility giảm so monitor làm bằng chứng đủ.

Last-A policy có T−1 action D rồi A cuối. Prefix trước settlement cuối phải khớp all-D; action cuối vẫn được train/send/aggregate/market settlement. Reputation thay đổi tại terminal không được biến thành penalty tương lai không được quan sát. H3 oracle không chứng minh tối ưu20, và fixed schedules20 không giải dynamic Nash.

### 16.3. Định nghĩa recovery và censoring

Resume depth r là state ngay trước action D đầu tiên sau attack burst. Kiểm điểm **peer chấm actor**, R_peer,actor, không đảo chiều thành actor chấm peer.

- Nếu đã ≥threshold tại r: `eligible_at_resumption`, không gọi là một lần phục hồi từ điểm thấp.
- Nếu còn thấp: tìm state đầu tiên sau D mà điểm đạt threshold; delay tính bằng số D transitions từ r.
- Không đạt trước horizon: `not_restored_by_horizon`, delay để null/right-censored, không gán vô hạn hoặc cho bằng horizon.
- Eligibility đạt tại terminal khác với đã nhận lại update ở một non-probe round. Ghi riêng terminal restoration, receipt sau eligibility và số receipt bị từ chối sau resume.
- Với K=1, mọi receipt đều là probe; không có non-probe receipt để quan sát, không suy “không phục hồi access” từ null ở metric này.

Mean delay chỉ báo trên những case đã phục hồi, cùng số censored/eligible-at-resumption để không che survivor bias. Đạt threshold một lần không chứng minh reputation ổn định vĩnh viễn hoặc mọi packet về sau được nhận.

Code: [mnist_access_sweep.py](../experiments/mnist_access_sweep.py). Candidate/data/protocol đã khóa cho sweep này nhưng chưa phải khóa final evaluation; official test vẫn chưa dùng.

### 16.4. Kết quả access và recovery

[Raw sweep](../experiments/results/mnist_access_sweep.json), [verification/tổng hợp](../experiments/results/mnist_access_sweep_verification.json), chi tiết bảng số ở roadmap §3.14. Tổng 18 root × (monitor + 4 K) = **90 cây H1–3, 900 rollout 20 bước**. K=1 khớp monitor; K=2 khớp reference ở những policies đã có. Không đổi source của candidate/credit trong sweep.

Non-IID clean Q_net giữ khoảng 87.54%, không có clean delivery bị chặn ở cả bốn K. Mean persistent-A gain J của actor 0 giảm khoảng 0.3950→0.1189 khi K=1→8, actor 1 giảm 0.5950→0.3325; vẫn có profitable cases ở cả hai actor. K=8 không được chọn làm final chỉ vì mean nhỏ hơn.

Sau burst ba A rồi D, qua 144 reciprocal actor/root/K cases: 140 phục hồi threshold eligibility trong window, 4 đã eligible ngay lúc quay lại D, không có right-censored case trong tập burst này. IID thường cần 2 D transitions; K=8/actor0 có một case tới 6. Non-IID conditional mean khoảng 1–2; không tăng đơn điệu theo K vì model/evidence cũng đổi theo history. Các K>1 đều có non-probe receipt sau eligibility ở các burst cases; K=1 không có non-probe round để đo.

Không gộp clean honest cases với burst switching: clean IID có 8 low-eligibility episodes/K, 7 restored và 1 right-censored tại horizon; actual delivery denials tăng từ 0 tới 10/360 khi K tăng 1→8. Clean non-IID không có episode dưới threshold trong sample này. Không suy zero false exclusion tổng quát từ 3 seed.

### 16.5. Counterexample không chỉ do probe: binary credit bỏ qua độ lớn đóng góp

Non-IID, seed 31, snapshot 5, K=8, sau profile đầu DA; actor 1 đang được node 0 chấm 0.5. Ở absolute round 6 (không phải probe), so cùng state/seed và peer luôn D:

| Actor 1 gửi | Realized validation improvement tại receiver | Binary credit | R_0,1 sau nhận | Q_net sau transition | Payoff actor 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D | 0.497907 | 1 | 0.75 | 81.85% | 0.511542 |
| A | 0.032585 | 1 | 0.75 | 76.95% | 0.557306 |

Cả hai giúp receiver so với model local giữ lại nên đều vượt epsilon. Nhưng A giúp ít hơn nhiều so D, giảm Q_net so cooperative counterfactual và tăng own payoff khoảng 0.04576, vẫn phục hồi eligibility giống D. Ở round 7 kế tiếp (cũng không phải probe), peer gửi lại cho actor 1. Remaining-two-step oracle value trước always-D cũng cao hơn khi chọn A, khoảng 1.13887 so 1.08928.

Hai counterfactual đã được retrain, tính loss model thật và replay raw hashes/quality/reputation. Đây là một finite seeded reachable-state counterexample, không phải tần suất dân số hoặc bằng chứng mọi A đều harmful. Receiver **không được biết bản D counterfactual** khi chấm runtime; clone này chỉ phục vụ phân tích. Immediate reputation giống nhau không bảo đảm mọi future state sẽ giống nhau.

### 16.6. Finite endgame và quyết định nghiên cứu tiếp

Ở cả 9 non-IID roots, hai actor đều có profitable last-A deviation trước peer always-D, với mọi K. Vì clean histories non-IID giống nhau giữa K, các bản này có tương quan/đồng nhất chứ không là repetitions độc lập. Mean final-settlement gain khoảng 0.02450 cho actor 0 và 0.04539 cho actor 1, đã trừ extra cost.

Với cùng prefix D tới lượt cuối:

\[
J_i(\text{last A})-J_i(\text{all D})
=\gamma^{T-1}\big(u_i^T(A,D)-u_i^T(D,D)\big).
\]

Sau T không còn payoff/action trong game hữu hạn hiện tại. Reputation hoặc future withholding chỉ thay ở terminal không thể trừ một continuation penalty không tồn tại. Nếu chênh lệch cuối dương thì all-D không phải best response trong phép so này. Không suy mọi state có A dominant, backward unraveling hoặc mọi equilibrium đều all-A.

**Quyết định sau access sweep:** có bằng chứng giảm một phần gain A nhờ access, chưa có full deterrence. Đã thử magnitude-sensitive earned access ở mục 17; không cấp defender oracle D, đổi utility/cost hoặc thêm fake terminal penalty. Mục tiêu vẫn là giảm poisoning và giữ quality, không ép all-D mọi round.

Trong các kết luận trên, “partial deterrence” chỉ là giảm gain của một số deviation schedules và thay đổi best response trong miền oracle đã nêu; **chưa đo giảm tần suất A của một learner local hoặc hội tụ EGT**. Không đổi nhãn fixed-policy rollout thành hành vi tối ưu/học được.

## 17. Earned-access ledger theo mức đóng góp

### 17.1. Đối chứng và đơn vị credit

Giữ realized-mix/service, các action/cost/market/horizon cũ. Hai ledger cùng accounting, initial balance=1, capacity=4 slots và price=1 slot/message:

- `binary`: một incoming update có positive realized validation improvement vượt epsilon kiếm 1 token.
- `magnitude`: token theo mức cải thiện tương đối của chính model được phục vụ, chuẩn hóa bằng public clean-development reference.

Binary ledger là ablation cần thiết: đổi từ EMA sang balance đã thay memory/accounting, không được quy mọi khác biệt so EMA cũ cho magnitude. Monitor không gate access và local-only all-D làm quality controls. Probe K=2/8 giữ recovery; không chọn K theo kết quả mới hoặc tăng economic cost.

Token không phải tiền/reward và không xuất hiện trong utility: vẫn market share trừ c_D/c_A. Nó chỉ điều khiển cơ hội nhận message. Một delivered packet có thể là A hoặc bị aggregation weight=0; **price không bảo đảm packet hữu ích**, không có oracle/refund theo validation kín của recipient. Risk provider gửi A nhưng vẫn tiêu quota recipient phải được giữ trong diễn giải.

### 17.2. Calibration không dùng evaluation-run logs

Đặt d_ij=max(0,L_i(ell_i)−L_i(served_i)), và:

\[
r_{ij}=\begin{cases}
d_{ij}/L_i(\ell_i), & d_{ij}>\epsilon,\ L_i(\ell_i)>0,\\
0, & \text{còn lại}.
\end{cases}
\]

Đây là fractional cross-entropy improvement, không phải accuracy gain. Với fold seed s, public scale b_{−s,t} là **median positive r** tại absolute action round t từ all-D monitor runs của các seed khác, pooled hai mode/actor/warmup roots. Không đọc market/audit metrics, không dùng bản D counterfactual của state hiện tại, không ép eval round đầu D.

\[
E_{ij}^{\rm binary}=\mathbf1[d_{ij}>\epsilon],
\qquad E_{ij}^{\rm magnitude}=r_{ij}/b_{-s,t}.
\]

Scale phải positive/finite và có sẵn cho mọi round yêu cầu; thiếu calibration thì fail rõ, không lấy evaluation data hoặc extrapolate fallback. Normalizer và sample counts/source seeds được lưu trước run, không chọn lại để ép deterrence.

Đây là **run/seed-excluded cross-fit**, không phải raw-image-disjoint: các world cùng lấy từ 60,000 MNIST train và có thể dùng trùng ảnh. Ba folds có calibration chồng nhau, không phải ba independent replications của cùng fixed b_t. Giả định deployment cần một public scalar schedule đã được calibration từ historical clean development data; không có central scorer hoặc truy cập validation peer tại runtime. Official test vẫn không dùng.

### 17.3. Quyền nhận, debit/credit và conservation

B_ij là balance do i giữ cho recipient j, tức giá trị đóng góp j đã đem lại cho i theo criterion local. Diagonal không dùng. Tại action round t:

\[
P_{i\to j}=\mathbf1[B_{ij}\geq1\ \lor\ t\bmod K=0],
\qquad S_{ij}=P_{i\to j}\min(B_{ij},1).
\]

Permissions chỉ dùng prior balance; ngay cả probe cũng tiêu phần balance đang có, phần thiếu được cấp free, không tạo debt. Sau khi message được giao và aggregate:

\[
B'_{ij}=\min(4,B_{ij}-S_{ij}+E_{ij}),
\qquad
F_{ij}=\max(0,B_{ij}-S_{ij}+E_{ij}-4).
\]

Nếu không nhận packet j→i thì E_ij=0, không tạo credit giả. B_ij vẫn có thể giảm vì i vừa gửi packet cho j; khác với missing-message freeze của EMA, đây là debit của outgoing service đã cấp, không phải phạt absence.

Logs giữ S/E/F để kiểm B'=B−S+E−F, 0≤B'≤4. Credit hiện tại chỉ tác động quyền nhận từ lượt sau. Initial one-slot endowment cho phép bootstrap từ cùng optimistic model snapshot; action đầu vẫn A hoặc D. Không có interest/expiry hoặc terminal liquidation thành utility.

Với enforcement bật, mỗi non-probe grant phải trả đủ 1 token, nên:

\[
N_{\rm nonprobe}(i\to j)\leq
\left\lfloor B_{ij}^{0}+\sum_t E_{ij}^{t}\right\rfloor .
\]

Probe spends và overflow chỉ siết bound; monitor không tuân eligibility nên không áp bound này cho monitor. Đây là access/accounting bound, không phải bảo đảm utility A âm hoặc global quality tăng.

### 17.4. Scope thực thi

Code [mnist_earned_access.py](../experiments/mnist_earned_access.py), hai credit modes × monitor/K2/K8 trên cùng 18 service roots; thêm local-only all-D rollout. Schedules recovery/endgame hai actor giữ từ access sweep; oracle vẫn H1–3 và fixed rollouts20, không optimal20/dynamic Nash.

State mới tên `balance`, không giả làm reputation xác suất hay điểm EMA. Tree/training/market/rollout dùng chung pipeline với reciprocity cũ; hooks state-key/initial-state không thay semantics baseline. Recovery dùng threshold một slot và field balance, vẫn tách eligibility/non-probe receipt/censoring. Historical artifacts không bị sửa checksum.

### 17.5. Kết quả: accounting đúng không đồng nghĩa incentive tốt hơn

[Raw ledger run](../experiments/results/mnist_earned_access.json), [verification/tổng hợp](../experiments/results/mnist_earned_access_verification.json), bảng số ở roadmap §3.15. Run có **108 cây H1–3 và 1,098 rollout20**: binary/magnitude × monitor/K2/K8 trên 18 roots, cộng 18 local-only all-D rollouts. Scale/price/cap đã chốt trước run, không đổi theo kết quả A.

Ở non-IID/K8, binary ledger có mean clean Q_net≈87.54%, gain persistent A của actor0≈0.15386 và actor1≈0.29216; magnitude ledger đạt Q_net≈87.40%, gain tương ứng≈0.23371/0.30367. Do đó magnitude không tốt hơn binary trong hai gain này. Local-only Q_net≈70.98%, nên magnitude vẫn giữ ích lợi FL, không phải hoàn toàn self-only; nhưng điều đó chưa đủ chứng minh răn đe.

Ở IID/K8, magnitude clean Q_net≈88.89% so binary≈89.31%, local-only≈87.52%. Sau ba A rồi D, magnitude có 9/18 quota-recovery cases chưa đạt lại threshold trong window; ở K2 là 10/18, còn binary không censored ở hai nhóm này. Đây là không phục hồi **đủ một slot**, không phải không bao giờ nhận probe packet hoặc bị blacklist vĩnh viễn.

Trước khi gate, trên monitor all-D, median raw magnitude credit là khoảng **0.247 token/packet IID** và **4.027 non-IID**, cùng giá một slot. 335/360 clean IID packets kiếm dưới một slot, còn 303/360 non-IID kiếm ít nhất một slot. Calibration pooled là một failure mode cụ thể của candidate: scale và market/quality value không tự tương đương. Không suy mọi magnitude-credit scheme đều thất bại.

### 17.6. Hai kiểm tra chống diễn giải sai

Ở model-state của counterexample mục 16.5, monitor diagnostic cho magnitude D≈2.551 tokens và A≈0.167, khác binary cùng1. Nhưng trong **enforced magnitude ledger**, sau prefix DA, balance đối ứng chỉ khoảng 0.224<1 nên packet D/A ở lượt kế đều không được gửi tới receiver0. Không gọi đây là một targeted fix: nó cũng cắt packet D hợp lệ ở state đó.

Capacity có thể lại xóa khác biệt về lượng. Non-IID seed7/snapshot0/K8, prefix DA/DD, actor1:

| Gửi | Raw tokens earned | Overflow | Next B_0,1 | Q_net | Payoff actor1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D | 9.71977 | 8.71977 | 4 | 66.40% | 0.45704 |
| A | 1.03693 | 0.03693 | 4 | 63.75% | 0.52949 |

Hai counterfactual được retrain và kiểm loss/credit/conservation. Đây là depth2 của oracle H3, minh họa cap saturation chứ không chứng minh một policy20 tối ưu. Không tăng price/cap hậu nghiệm chỉ để tách riêng điểm này: điều đó còn ảnh hưởng clean IID đang thiếu quota.

### 17.7. Quyết định sau ledger pilot

Giữ magnitude ledger đã thử làm **negative-control candidate**, không chọn nó thay binary/EMA hoặc tuyên bố incentive compatibility. Không đưa tokens vào utility hay bổ sung terminal penalty để cứu kết quả.

Bước tiếp ưu tiên chốt miền policy và learner dài hạn cục bộ trên các mechanism đã đóng băng, đo A-rate/switching/quality và finite-endgame behavior thực sự. Các fixed schedules hiện tại chỉ đưa ra deviation evidence; chúng chưa cho biết agent học sẽ giảm A. Nếu quay lại thiết kế ledger, cần giải quyết calibration context/liquidity và saturation bằng điều kiện rõ, không tiếp tục siết scalar theo score đẹp.

Evaluation policy tiếp theo phải tách A-rate trên mọi round khỏi A-rate khi có outgoing delivery opportunity. Cần baseline local “attack-if-can-send” dựa trên own link state/public clock, thay vì chỉ always-A ngay cả khi packet không được gửi; baseline này chưa được chạy trong ledger pilot. Không được gọi giảm cơ hội gửi là bằng chứng agent đã tự nguyện hợp tác.

## 18. Local episodic policy learner trên mechanism đóng băng

### 18.1. Protocol trước khi xem kết quả

Giữ realized-mix/service, attack scale=5, market/cost/gamma và dữ liệu như các run trước. So bốn conditions đã có: monitor, EMA/K8, binary ledger/K8, magnitude ledger/K8; không retune mechanism. Chọn trước model snapshot round5 cho cả IID/non-IID và ba seed7/19/31, không chọn root theo dấu gain. Mỗi episode dài 20 action/settlement; link state reset optimistic như các đối chứng cũ.

Policy là Bernoulli logistic, theta ban đầu bằng 0 nên p(A)=0.5, không action mask hoặc entropy/reward bonus. Features gồm bias; progress, final-round flag và probe công khai; outgoing eligibility từ **own evaluator row**; own link value chia capacity; own receipt/action trước đó và own market share. History tại snapshot được zero-initialize với at-start flag riêng. Không dùng peer row, peer private quality/action label, market quality vector, audit, current D counterfactual hoặc future simulator state.

Từng seat train riêng trước peer luôn D, bằng own trajectory/reward qua các episode reset độc lập; không cấp reward của peer. Sau đó theta đóng băng để evaluate actor0-vs-D, actor1-vs-D và hai policy đã train riêng cùng deploy. Joint deployment này **không phải self-play co-learning**. Đây là repeated-task development learner, chưa là adaptation online của một DFL trajectory liên tục.

### 18.2. Update dài hạn và đối chứng một bước

Với x_t là observation features, p_t=sigmoid(theta^T x_t), score_t=(a_t−p_t)x_t. Hai updates đều dùng đúng utility market-share−cost và trọng số gamma^t/Z, Z=sum_t gamma^t:

- `return`: target G_t=sum_{k=t}^{T−1} gamma^{k−t} u_k; gradient trung bình score_t (G_t−b_t) gamma^t/Z.
- `myopic`: target chỉ u_t, bỏ ảnh hưởng tương lai. Đây là **semi-gradient**, không phải gradient đầy đủ của episodic J; không gọi nó gamma=0 của cùng objective rồi bỏ mọi action sau đầu episode.

b_t là mean target của các episode **khác** trong cùng batch (leave-one-out), độc lập với episode đang lấy score. Return estimator là score-function gradient của J/Z trong seeded environment, không cần backprop qua model, discrete action hoặc endogenous observation.

Default chốt trước run: 16 updates × 4 episode/seat/method, learning rate4, một learning-seed101, hai methods và eval16 action streams chưa dùng train. Không normalize advantage riêng theo method hoặc thêm terminal penalty. Đây là pilot nhỏ, không bảo đảm đủ train hoặc policy tối ưu.

### 18.3. Đối chứng, randomness và metric

Controls gồm D/D, A/D, D/A, attack-if-can-send ở từng seat và cả hai, A/A, iid-half ở từng seat và cả hai, cùng local-only D/D. Attack-if-can-send chỉ dùng own outgoing eligibility đã xác định trước action; nó vẫn có thể A khi packet được gửi, không biết private grade của peer.

Action RNG tách theo actor và phase train/eval; cùng uniforms qua mechanism/method để paired comparison. SGD giữ lịch seed lịch sử theo world/node/absolute round, nên evaluation chỉ tích phân action randomness có điều kiện trên seeded SGD world; không phải held-out-image hoặc full stochastic-training evaluation. Train chỉ tính market feedback, **không evaluate audit**; audit dành cho reporting ở frozen-policy eval. Official test không dùng.

Báo J/gain so D/D trong cùng condition và so initial iid-half control; Q_net/Q_min/outside; A-rate toàn round và **A_count ở outgoing opportunities / số opportunities**; switching chỉ giữa hai action thực, terminal/preterminal riêng. Không average các conditional fractions undefined hoặc coi ít opportunities là tự nguyện hợp tác.

Baseline attack-if-can-send phải được kiểm bằng model replay so always-A: khi outgoing không được giao, đổi A thành D không được đổi model/link trajectory; khác biệt utility chỉ có thể là attack-cost tiết kiệm. Không được suy nó đã chạy hoặc learner đã học quy tắc đó trước khi có output.

Implementation [mnist_local_policy.py](../experiments/mnist_local_policy.py), [independent verifier](../experiments/verify_mnist_local_policy.py), [raw run](../experiments/results/mnist_local_policy.json), [verification](../experiments/results/mnist_local_policy_verification.json). Bảng số tại [roadmap §3.16](ROADMAP.md#316-local-episodic-learner--protocol-đóng-băng).

### 18.4. Kết quả và giới hạn behavior evidence

Main run đủ 24 world/condition cases, 96 seat/method training runs và 1,536 gradient updates: **6,144 train episodes + 3,648 evaluation episodes**, mỗi episode20 settlements. Một learning-seed101; 16 eval action draws/world/policy, không phải 16 data replications. Không đổi predeclared LR/batch/update count sau khi xem kết quả.

Với non-IID joint deployment, return learner có opportunity-conditioned A-rate thấp hơn myopic ở cả bốn conditions, nhưng vẫn cao hơn initial p(A)=0.5. Binary/K8 có return A-rate thấp hơn monitor và quality tốt hơn monitor; đây là **partial behavioral suppression trong pilot**, không phải A mất lợi thế. Với IID, direction không đồng đều: return giảm A ở EMA nhưng tăng ở ba conditions còn lại.

Ở unilateral-vs-D/non-IID, mean gain J của learned return policy so D/D vẫn dương cho cả hai seat ở mọi condition. Mean không đại diện mọi world: actor0 EMA và binary chỉ profitable ở một trong ba world, actor1 profitable ở cả ba. Không kết luận policy tối ưu, best response, hoặc tất cả player-state chọn A.

Joint policy Q_net thấp hơn initial iid-half joint control ở cả bốn conditions, dù return thường ít A hơn myopic trong non-IID. Các policy train riêng trước D không được tối ưu trước nhau; deployment này không phải equilibrium và không chứng minh simultaneous co-learning sẽ tới cùng kết quả. Finite logistic feature family và LR chung không bảo đảm cùng độ khó optimization giữa các state representations.

### 18.5. Opportunity baseline chống đếm nhầm giảm A

Ở binary/K8/non-IID/actor0 trước D, attack-if-can-send chỉ chọn A ở 45% round nhưng vẫn A tại 100% outgoing opportunities. Always-A chọn A mọi round, song hai policy có cùng actual sent-A packets và cùng model/link/weight/quality trajectory. Utility của attack-if-can-send cao hơn đúng phần attack-cost tránh khi packet không được giao. Vì vậy không thể gọi chênh A-rate thô này là răn đe hoặc giảm poisoning vật lý.

72 paired paths trên tất cả mode/condition và hai seat (cộng joint) kiểm được đồng nhất model/evidence và cost-only difference. Đây là baseline local đã chạy, không full-state oracle: nó chỉ cần own outgoing eligibility. Learners không được hard-mask theo baseline; A khi không gửi được vẫn được sample và charge cost bình thường.

### 18.6. Kiểm chứng và quyết định

26 regression tests trên game/reciprocity/access/ledger/local policy; thêm finite-difference expected-J check bằng enumeration stateful two-step trajectories và các independent episode pairs, delayed payoff đảo chiều so myopic, hidden-peer-row intervention, zero-opportunity denominator và terminal settlement/switching.

Verifier recompute riêng 1,536 LOO gradient batches (max absolute discrepancy≈2.48e−16), 267,840 policy decisions gồm RNG replay và observation projection, 72,960 market settlements, 90 calibration medians và sáu disjoint within-world split manifests. 96 historical controls giữ đúng mọi model/state/evidence field. Retrain hai learned joint episodes bằng legacy rollout; audit index cố tình out-of-bounds khi chạy training path vẫn cho trace y hệt frozen evaluation, model roots bất biến.

Giữ các mechanism làm đối chứng, không chọn magnitude hoặc binary thành final defense từ bảng đẹp. Bước tiếp ưu tiên uncertainty/robustness của learner (nhiều learning seeds, initial-p/horizon/world sensitivity), sau đó mới xét joint co-learning và policy-level EGT với fitness từ episodic J. Không gọi policy-gradient này là EGT hoặc dùng một frozen payoff matrix cho toàn training. Official test tiếp tục dành cho protocol đã khóa.

## 19. Robustness của learner và chuyển policy sang world mới

### 19.1. Protocol đóng băng trước main run

Giữ nguyên bốn mechanisms, architecture/data/attack/cost/utility/gamma, root5 và optimizer của §18 (16×4 episodes/seat, LR4). Dùng OFAT, không full factorial:

| Trục | Values | Methods |
| --- | --- | --- |
| Learning RNG | 101, 211, 307; initial-p=.5, T20 | myopic và return |
| Initial p(A) | .2, .5, .8; learning seed101, T20 | return |
| Horizon | 10, 20, 25; learning seed101, initial-p=.5 | return |

Baseline training seed101/p=.5/T20 được lấy nguyên từ artifact §18, **không train lại**; evaluation cũ không reuse. Mọi policy/control mới evaluate trên cùng action RNG seed901/phase2, khác phase0 của training và khác evaluation §18. Nhờ vậy range qua learning seeds không bị trộn thêm thay đổi eval-seed. Không đổi số update/LR theo kết quả, không đưa token hoặc terminal penalty vào utility.

Initial probability chỉ đổi bias logit của cùng logistic policy; mọi feature coefficient khác khởi tạo0. Existing §18 CLI truyền .5 rõ, semantics baseline không đổi. Historical artifact/source hashes giữ nguyên; source mới ghi checksum riêng, không ký lại output cũ bằng source mới.
Verifier §18 có thể replay archived source với digest đúng của baseline trước cutover initializer. Snapshot source được checksum trước khi execute; không bỏ qua source mismatch hoặc rewrite metadata cũ để cho check pass.


### 19.2. Horizon, metric và uncertainty

Probe giữ absolute-round phase và gamma=.95. Public progress/final-round flag phụ thuộc T; T25 dùng đúng calibration đã có tới action round29, thiếu scale phải fail. Horizon sensitivity đổi cả chiều dài learning trajectory/endgame/probe exposure, không chỉ patience và không được diễn giải thành infinite-horizon result. Số episodes/updates bằng nhau nhưng số transitions/compute không bằng nhau qua T; không phải matched-step-budget comparison. Gain so D/D phải cùng T; thêm gain/Z, Z=sum gamma^t, để không nhầm độ dài return với incentive strength.

Mọi condition giữ D/D và matched initial-policy controls ở cả hai unilateral seats và joint deployment. Báo raw pooled sent-A/opportunity, all-round rate, J/gain, quality, switching/terminal; range qua ba learning seeds trên từng cohort và paired method/mechanism contrasts. Sampling unit cho action Monte Carlo là **episode**, không individual rounds hoặc node-actions; conditional MC error không phải CI trên general population/data worlds. Ba worlds cũ có thể dùng trùng ảnh, không coi thousands of rollouts là thousands of independent data replications.
MC SE của conditional A dùng delta approximation với raw sent/opportunity counts gộp hai node theo từng episode; không coi hai node là independent samples. Common eval draws giúp pairing, không loại hết finite-MC error; range qua learning seeds là descriptive range của estimated policy outcomes, không phải exact variance của true expected values.


### 19.3. Zero-shot policy transfer, không phải image holdout

Định trước mappings7→43,19→59,31→71, cùng IID/non-IID mode. New target worlds dùng cùng split recipe và clean-uniform warmup5; source snapshots reuse, không retrain. Chỉ chuyển learned policy theta của base cohort, **không chuyển model hoặc ledger/reputation**, không update policy trên target feedback. Evaluate unilateral-vs-D và joint frozen deployment với matched target D/D/initial controls.

Target seeds không xuất hiện trong policy training hoặc bất kỳ calibration reference nào. **Giữ nguyên public scalar table của source fold khi chuyển target**, tránh lẫn data-world shift với normalizer shift. World mới vẫn lấy từ official MNIST train nên ảnh có thể overlap qua worlds: đây là world-seed transfer, không image-independent generalization/final test. Own target share/link state được quan sát lúc acting như thông tin local hợp lệ; không dùng target audit chọn policy.

### 19.4. Reuse và kiểm chứng

Implementation [mnist_policy_sensitivity.py](../experiments/mnist_policy_sensitivity.py), [checkpoint consolidator](../experiments/merge_mnist_policy_sensitivity.py). Chuẩn bị roots/manifests riêng trước condition shards; root source giữ nguyên hash, chỉ tạo sáu target snapshots. Resume chỉ chạy missing source/mode/condition cases; merge từ checkpoint/subset khác nhau phải cùng source/protocol, không duplicate và đủ cohort mới phát hành master. Evaluation logs giữ actor traces, model hashes, states/permissions/weights và market/audit/utility; không lặp lại CE-grid diagnostics của studies đã kiểm. Không tuyên bố re-audit credit conservation từ lean logs.

Đã kiểm 6,144 new LOO gradient batches (max discrepancy≈2.31e−16), 1,286,160 policy decisions và 405,000 market settlements; 12 split manifests, 72 historical retained-record D/D prefixes, 96 training runs reuse. Physical replay gồm initial-p=.2 binary/non-IID, T25 EMA/IID và zero-shot magnitude/target43; actor traces/audit exclusion và retained model/state/quality records khớp. Target warmup reconstruction và neutral-initializer legacy first batch đều pass. Consolidator từ chối duplicate/incomplete checkpoints mà không xuất partial master.

### 19.5. Robustness có điều kiện, không phải một A-rate hội tụ

[Raw master](../experiments/results/mnist_policy_sensitivity.json), [verification/paired-control summaries](../experiments/results/mnist_policy_sensitivity_verification.json), [bảng đầy đủ tại roadmap §3.17](ROADMAP.md#317-learner-robustness-và-zero-shot-world-transfer--protocol). Đủ 24 source/24 target cases, gồm 24,576 new train và 20,832 eval episodes; interrupted checkpoints được giữ nguyên và chỉ bù missing cases. Không retune scalar/learner sau khi xem output.

Trong non-IID/T20/p=.5, return learner ít A/opportunity hơn myopic ở cả ba learning seeds của cả bốn conditions. Binary có mean A/opportunity thấp hơn và quality cao hơn monitor trong cohort này, nhưng các learned deviations vẫn có positive mean utility trước D. IID chỉ EMA giảm A; không có kết luận dài hạn luôn thúc đẩy hợp tác.

Initial-p làm thay đổi outcome lớn hơn range qua learning RNG. Nó đổi exploration và gradient conditioning; finite training không chứng minh các giá trị cuối là attractors hoặc equilibrium. Không chọn prior thiên D để claim defense đã làm agent hợp tác.

### 19.6. Horizon/endpoint quality và denominator của endgame

Binary/non-IID có A-rate gần nhau ở T20/T25 nhưng final quality giảm mạnh ở T25. Trong chính saved T25 trajectory, quality tăng tại probe action-round24 rồi giảm cùng useful-sharing/access tới action-round29. Không dùng endpoint ở probe làm bằng chứng quality ổn định; đây là diagnostic observed trajectory, không xác định riêng causal effect của access/probe.

Terminal opportunity counts có thể rất nhỏ hoặc bằng0. Những conditional rates 100% với ít admitted packets không chứng minh all-A optimum; denominator0 phải giữ null/NA, không gán rate0 hoặc tự nguyện D. T20 kết thúc ở probe, T10/T25 không; đây là một lý do phải báo eligibility/delivery và quality theo thời gian thay vì chỉ final A-rate.

J so D/D được ghép cùng horizon; J/Z chỉ là normalization phục vụ so sánh, không thay utility hoặc thêm reward. Raw gain tăng khi T dài hơn không tự có nghĩa poisoning incentive mạnh hơn. Số transitions/compute khác qua T và progress/final flag cũng khác; không suy gamma/patience-only effect.

### 19.7. New-world policy transfer và quyết định nghiên cứu

Transfer giữ nguyên theta và source scalar schedule, chỉ dùng own target feedback/link state lúc acting; zero-shot ở mức policy parameters không có nghĩa không đọc target feedback hoặc không train target model bằng SGD. Target seeds không vào policy training/calibration, nhưng raw images có thể overlap qua worlds.

Non-IID target means vẫn có partial A suppression, cùng positive estimated unilateral gains ở cả hai actor/mọi condition. Joint quality dưới matched target D/D và initial-policy controls; không so absolute Q giữa hai world rồi gọi đó là generalization improvement. EMA/binary có sign reversal giữa một số target worlds, nên mean không phải universal statement về mọi node/state. IID EMA transferred-policy means kém D không chứng minh không tồn tại profitable deviation: learner/transfer failure không được gán thành incentive compatibility.

**Quyết định:** không promote mechanism từ một RNG hoặc một endpoint đẹp. Ưu tiên đóng băng episode-level policy game với full-trajectory J, đánh giá policy-level selection hoặc joint co-learning như hai processes có scope riêng. Mọi policy vẫn chọn A/D theo local observation, không gán permanent attacker/defender types. Một payoff matrix chỉ có ý nghĩa cho fixed episode-reset protocol/world/initial-state distribution đã công bố, không đại diện continuous DFL state drift. Chưa có EGT convergence, ESS, dynamic Nash hoặc final defense; official test tiếp tục giữ riêng.


## 20. Episode-reset policy game và invasion có scope

### 20.1. Strategy là local rule, không phải identity A/D

Đóng băng bốn strategy bundles: **D**, **attack-if-send**, **myopic**, **return**. Hai learned bundles lấy đúng coefficients của từng seat từ baseline learning-seed101, initial-p=.5, T20; mỗi seat đã train riêng trước D. Bundle định nghĩa policy theo world và seat, không phải một universal theta chuyển qua mọi node. Mỗi policy vẫn quyết định A/D từng round từ observation local ở §18; không đọc opponent policy label, peer quality hoặc audit. Không train lại hay cập nhật theta giữa encounters.

Trong mỗi mode/condition, game lấy đều ba development worlds7/19/31, cùng model root sau năm clean-uniform warmup rounds. Mỗi episode reset model, optimistic link state và local history; T20/gamma=.95, absolute clock/probe giữ nguyên. Fitness là **J_i=sum_t gamma^t u_i(t)** của cả trajectory, không phải final accuracy, immediate reward của myopic training hoặc balance tokens. Không đổi utility/cost, attack, scalar table hoặc thêm terminal shaping.

### 20.2. Hai seat cố định và auxiliary random-seat game

Đặt U0[i,j] là expected J của seat0 khi seat0 dùng bundle i và seat1 dùng j; U1[i,j] là payoff của seat1 trong **cùng profile đó**. Hai ma trận giữ riêng, không giả định hai node/data giống nhau. Seat0 deviation thay hàng; seat1 deviation thay cột. Pure Nash/regret chỉ trong bốn frozen bundles, từ plug-in empirical means của fixed reset distribution; không phải tối ưu trên tất cả local policies.

Để khảo sát một population các bundles được gán seat ngẫu nhiên mỗi encounter, định nghĩa riêng:

$$M_{ij}=\tfrac12[U0_{ij}+U1_{ji}].$$

Term thứ hai dùng **focal bundle i tại seat1**, không dùng U1[i,j] là payoff của opponent. Gán seat ngẫu nhiên là giả định của auxiliary selection game, không thay fixed identities trong physical DFL. Không lấy ma trận đối xứng (M+Mᵀ)/2 hoặc tráo payoff đối thủ vào focal fitness.

Trong replicator của auxiliary game, rare-mutant growth quanh homogeneous resident j là M[i,j]−M[j,j]. Đây là local boundary diagnostic của **episode-reset bundle population**. Positive empirical growth cho biết invasion direction trong game hạn chế này; không phải observed adaptation, fixation probability, dynamic Nash hoặc ESS của mạng DFL liên tục. Không kết luận sign chắc chắn khi action Monte Carlo gần0.

### 20.3. Matchup completion, reuse và uncertainty

Mỗi world chạy đủ16 ordered profiles. Reuse nguyên bốn deterministic D/attack-if-send controls (một exact trajectory/profile) và sáu learned unilateral/joint phase2 profiles từ §19. Sáu cross-policy profiles còn thiếu được evaluate mới với16 common action draws, eval-seed901/phase2, cùng actor-specific streams; không lấy evaluation phase1 của learned baseline. Deterministic controls không phụ thuộc action RNG và không nhân bản thành16 independent observations.

Fixed-world mixture cho mỗi world trọng số1/3, không gộp episode counts làm deterministic profile nhẹ hơn stochastic profile. Paired action-MC SE dùng draw index chung giữa profiles và hai seat; sampling unit là episode. Báo per-world deviation/invasion signs riêng, không dùng ba source worlds làm population CI. Không dùng target worlds hoặc official test cho game này.

Runner [mnist_policy_game.py](../experiments/mnist_policy_game.py) checkpoint sau mỗi completed cell, resume chỉ giữ inputs/source/config khớp. [Independent verifier](../experiments/verify_mnist_policy_game.py) kiểm đủ cells, reuse equality, local observation/action RNG, utility settlement, full-J và role-swap; legacy model replay cùng audit-exclusion smoke kiểm changed cross-policy path. Historical artifacts/source hashes không bị ghi lại.


### 20.4. Kết quả: restricted equilibrium vẫn poisoning

[Raw game](../experiments/results/mnist_policy_game.json), [verified matrices/invasion/curves](../experiments/results/mnist_policy_game_verification.json), [bảng tại roadmap §3.18](ROADMAP.md#318-frozen-episode-policy-game--protocol). Hoàn tất384 cells/24 cases,4,704 episodes gồm2,304 newly evaluated và2,400 reused; không training mới.33 regression tests pass; independent arithmetic kiểm188,160 decisions/94,080 settlements. Hai legacy20-step cross-policy replays và audit-exclusion actor traces exact, roots bất biến.

Trong mọi mode/condition mean game và từng world, (attack-if-send,attack-if-send) là unique pure Nash của **restricted plug-in empirical matrices**. Trong auxiliary random-seat game, attack-if-send invade D và return với positive estimated growth ở từng world; D/myopic/return đều có negative estimated growth khi invade attack-if-send. Không có lý do từ bảng này để gọi mechanisms đã răn đe poisoning, hoặc gọi learned return là equilibrium behavior.

Đây không phải strict dominance trên mọi opponent/seat: có counterexamples mà return tốt hơn attack-if-send trước D. Role-averaged invasion cũng không có nghĩa mọi seat hưởng lợi; IID/EMA có near-zero negative seat0 mean nhưng positive seat1 mean cho attack-if-send deviation trước return. Không suy global replicator convergence, absence of mixed equilibria hoặc exact expected-payoff signs từ finite action samples.

Quality khác incentive: attack-if-send poison100% permitted outgoing opportunities, dù total-round A thấp khi access bị khóa. EMA/binary có final Q cao hơn monitor tại probe endpoint nhưng **time-mean Q qua20 settlements thấp hơn monitor**, ở cả attack-if/attack-if và return/return non-IID profiles. Chỉ endpoint đẹp không chứng minh uniform damage reduction. Giữ utility/cost và mechanisms nguyên trạng; bước tiếp là scoped sensitivities/policy selection hoặc joint adaptation, không sửa scalar để ép all-D.


## 21. Đề xuất utility mới: ký quỹ tuân thủ có audit

### 21.1. Quyết định và phần context thay đổi

Đề xuất **market profit + trách nhiệm hợp đồng được thực thi bằng collateral**, thay vì thưởng trực tiếp vì chọn D hoặc chỉ cộng điểm reputation. Đây là nhánh mới theo yêu cầu xem lại reward/payoff; không sửa utility hay ký lại kết quả §§10–20.

Mỗi node vẫn là cùng một player, mỗi round chọn D/A và giữ clean local model. DFL vẫn không có central aggregator; W vẫn dùng validation riêng. **Giả định thêm** là các thành viên có identity cố định đã ký một hợp đồng trao đổi update, có collateral có thể bị thu và một primitive kiểm chứng thực thi canonical training. Financial settlement/proof verification là trust/enforcement substrate mới, không được giấu nó dưới tên “local-only W”.

Không đề xuất “cộng reward theo số neighbor”, “thưởng clean label” hoặc “tăng attack cost tới khi D thắng”. Chỉ có **sự kiện vi phạm có thể kiểm chứng** mới làm mất tài sản thực. Không lấy validation loss cao của receiver làm bằng chứng node đã chọn A: honest non-IID có thể bị đánh nhầm.

**Trạng thái:** proposal kèm sufficient theorem và numerical feasibility certificate; chưa có prover/escrow triển khai trong MNIST runner, chưa có measured audit accuracy/cost, chưa claim voluntary adoption hoặc end-to-end deployable deterrence.

### 21.2. Hợp đồng, observation và timing

**Canonical obligation.** Trước episode, commit training data, initial serving/model state, optimizer/SGD recipe và seed schedule. Mỗi round, nghĩa vụ gửi là đúng clean local result ell_i từ serving input được protocol cho phép. Chứng cứ phải ràng buộc cả input-state chain và **hash của payload thực sự đã gửi**; chứng minh một alternate clean model rồi gửi model khác không hợp lệ. Commitment data không chứng minh data trung thực: data poisoning/backdoor qua training hợp lệ vẫn ngoài guarantee này.

**Audit sau commit.** Sender ký/commit payload trước; sau đó mới có audit coin không đoán được từ history lúc chọn action. Audited sender cung cấp proof thực thi canonical relation với dữ liệu đã commit. Một audited noncanonical payload, proof sai hoặc refusal theo deadline có thể dẫn tới certified fault. Auditor/settlement kiểm statement đã công bố, không đọc action label từ simulator. Mere signature hoặc checkpoint do sender tự khai không chứng minh training đúng.

Receiver chỉ cần message/commitments/proof/receipt thuộc quan hệ của nó, public protocol và own local state. Không cấp peer true quality, private data, audit MNIST population hoặc counterfactual market return cho policy. Financial verifier phải xác nhận được payload availability/receipt và deadline; một complaint không có signed/binding evidence không được tự động lấy tiền sender. Censorship, giả receipt, equivocation, Sybil và audit-coin manipulation đều là prerequisite của enforcement, không phải đã được giải bằng utility formula.

**History/connectivity.** Ledger tối thiểu có `(identity, active, collateral, payload_commitment, audit_result)`. Trước certified fault: dùng realized-mix/monitor receive rule, không thêm EMA/quota hay ép useful weight dương. Sau first certified fault: thu toàn bộ bond, đặt active=0, không cho identity đó gửi thêm update được protocol chấp nhận trong episode; node vẫn giữ model riêng và vẫn có thể cạnh tranh khách hàng. Không có K8 probe tự động khôi phục hoặc account reset để tránh mất bond. Dù audit có độ trễ làm một poisoned update đã ảnh hưởng model, proof bên dưới vẫn tính toàn bộ khả năng thu lợi còn lại.

**Funding/settlement.** Bond được bảo đảm trước khi gửi, không phải lời hứa sẽ trả fine sau khi nhận market gain. Thu bond **một lần** tại certified fault; sau đó không phát sinh nghĩa vụ fine mới không có collateral. Fines vào neutral settlement/insurance account, không trả bounty cho strategic peers; không có external subsidy cho D. Audit/funding fee cho episode được commit/prepaid, không được né bằng cách cố tình bị loại. Không tính mất receive access là một utility loss dương nếu chưa chứng minh được giá trị của access.

Audit cuối và collateral liability vẫn phải được quyết toán trong final economic settlement. Nếu audit thực sự trễ d bước thì phải discount fine theo gamma^d và tính financial tail; không được cắt hết liability ở T rồi nói history răn đe final action. Unlock principal không phải reward: collateral vẫn là tài sản của owner cho tới khi bị tịch thu. Đây là **economic-profit/asset-loss utility**, không được vừa ghi toàn bộ deposit là cost vừa cộng toàn bộ refund thành performance reward. Nếu mô hình hóa cash-flow utility thay cho asset loss, phải định giá đúng lúc nộp/thu/hoàn tiền và kiểm lại IC/participation.

### 21.3. Utility đề xuất và assumptions của theorem

Chuẩn hóa tiền theo v_i*M như utility cũ. Với F_i là bond, Y_i^t=1 đúng khi first certified fault được quyết toán cho action t, đề xuất:

$$
\widetilde J_i(\pi)=
\mathbb E\!\left[\sum_{t=0}^{T-1}\gamma^t
\left(\sigma_i^{t+1}-b_i^t-m_i^t\,1\{a_i^t=A\}
-F_iY_i^t\right)\right]-K_i.
$$

K_i là fixed prepaid audit/funding cost thực của hợp đồng, cùng cho mọi policy trong hợp đồng; **không phải** reward mới hoặc cost vật lý của A. Financial loss F_i*Y là mất collateral có thể cưỡng chế. Principal ban đầu/được unlock không tạo thu nhập trong profit accounting này.

Các assumptions cần để chứng minh:

1. `0<gamma<=1`, T hữu hạn; mọi market share thuộc [0,1]. Common base cost schedule không giảm vì bị loại; attack increment m>=0. Không có khoản thu khác phụ thuộc strategy ngoài các khoản đã ghi. K_i không thể được tiết kiệm bằng deliberate default.
2. Một sender tuân thủ canonical training không bị certified fault: completeness/availability đúng, không có false-positive seizure.
3. Tại **mọi active history**, mỗi noncanonical payload có conditional probability bị certified fault ít nhất p_min>0, kể cả policy adaptive. Commit diễn ra trước audit randomness; soundness error và collusion/censorship phải được đưa vào p_min.
4. Bond còn nguyên trước first certified fault, có thể thu tại settlement; sau fault không còn admitted outgoing messages. Identity không được đổi để né hợp đồng. Node vẫn có mọi lợi ích cạnh tranh sau khi bị loại, nhưng chúng vẫn nằm trong share bound của assumption1.
5. Proof ràng buộc private input/state/data/seed và actual payload như §21.2; chỉ kiểm public hash do sender tùy ý chọn không đủ.

Đây là assumptions bổ sung thật, không phải thông tin đã có từ local validation. Một candidate kỹ thuật là verifiable computation/zero-knowledge training hoặc attested execution với trust model riêng. [zkDL](https://eprint.iacr.org/2023/1174) nghiên cứu proofs của deep-learning training; điều đó **không** chứng minh NumPy/IEEE arithmetic, state chain, data availability, ledger hoặc latency của runner này đã tương thích. [Attacks on Proof-of-Learning](https://arxiv.org/abs/2108.09454) là lý do không thay sound execution proof bằng heuristic checkpoint matching. Chưa chọn backend hoặc gán chi phí bằng0.

### 21.4. Theorem: deterrence của mọi profitable message deviation sau entry

Đặt discount mass của h rounds còn lại:

$$Z_h=\sum_{k=0}^{h-1}\gamma^k.$$

Với eta>0, chọn bond **được fund thật** thỏa:

$$\boxed{p_{\min}F_i\ge Z_T+\eta.}$$

**Kết luận có điều kiện.** Trong action/game domain đã nêu, gửi canonical update ở mọi round là optimal trước mọi opponent continuation; mọi policy có positive probability gửi một noncanonical message khi còn active có expected return thấp hơn policy luôn tuân thủ. Kết luận áp dụng cả last action; không giả định mất connectivity có lợi ích âm cho attacker.

**Proof.** Ghép hai trajectories tới first noncanonical send tại stopping time tau. Trước tau, physical/contract histories giống nhau; nếu có action nhãn A tạo đúng payload D thì chỉ thêm nonnegative cost, không làm tăng gain trước tau. Condition trên history đó, còn h=T−tau rounds.

- Bỏ qua mọi fine, lợi ích operating-profit tối đa của toàn bộ deviation continuation so với luôn D không vượt Z_h: mỗi round share tăng không quá1, common b triệt tiêu, m>=0 không tăng gain. Bound này đã bao gồm damage lan tỏa, thay đổi model riêng, opponent reactions và mọi lợi ích sau khi bị loại.
- Luôn D không mất bond theo assumption2. Ngay first noncanonical send, expected collateral loss của deviation ít nhất p_min*F_i; nếu lần đó không bị phát hiện mà bị phát hiện sau, loss thêm chỉ làm deviation kém hơn.
- Do đó conditional full-continuation gain không vượt `Z_h−p_min*F_i <= −eta`. K_i đã prepaid như nhau nên triệt tiêu; không cộng tùy ý một “connectivity reward”.

Lấy expectation qua stopping histories:

$$
\widetilde J_i(\pi_i,\pi_{-i})-
\widetilde J_i(D,\pi_{-i})
\le -\eta\,\mathbb E[\gamma^\tau\,1\{\tau<T\}]<0
$$

nếu có noncanonical send với probability dương. Đây là first-deviation proof cho toàn bộ policy continuation, không phải chỉ stage payoff hoặc một frozen policy set. Ở last action h=1, bound vẫn dùng `1−p_min*F_i<0`; không có finite-horizon unraveling do liability biến mất.

**Giới hạn kết luận.** Nếu một action mang nhãn A nhưng tạo đúng canonical payload, không có certified violation: physical message giống D và chỉ còn nonnegative extra cost; với m=0 có thể indifferent. Theorem không khẳng định mọi policy parameter hóa khác D đều strict-worse, cũng không cover dedicated Byzantine utility thưởng damage, data poisoning, forged execution substrates hoặc free entry/exit. Không tự suy ESS, EGT convergence hoặc model-quality guarantee từ financial incentive compatibility.

### 21.5. History/connectivity làm gì, và không làm gì?

Nếu cùng broadcast payload được d neighbor audit độc lập với rates a_ij, sau khi sender commit, và soundness failure conditional trên audit không quá epsilon:

$$
p_i\ge(1-\epsilon)\left[1-\prod_{j=1}^{d}(1-a_{ij})\right].
$$

Connectivity có thể giảm required collateral **qua coverage audit**, không phải vì mỗi cạnh tự sinh reward. Formula này không đúng nếu audit coins tương quan hoặc một targeted payload chỉ có auditor không kiểm được nó; cần lower bound theo từng actual recipient/payload. Degree0 không có deterrence certificate từ audit, nhưng cũng không có admitted outgoing channel để poison theo game hiện tại. History active/slashed giúp enforce funding/identity; heuristic reputation không thay được certified evidence.

Nếu chỉ dùng delayed exclusion, tại final action không còn future access để mất. Nếu **toàn bộ contractible signal history tới settlement** có cùng phân phối dưới deviation và compliance nhưng deviation có positive base gain, mọi transfer/history rule chỉ dựa vào history đó có cùng expected sanction: không tự tạo được deterrence. Ở terminal, chỉ cần current signal không phân biệt được; ở nhiều rounds, current d=f riêng lẻ chưa chứng minh toàn bộ future histories cũng không phân biệt được. Đây là information constraint, không phải thiếu một hệ số lambda lớn.

Nếu có false positives, proof §21.4 **không còn áp dụng nguyên xi**. Ở terminal, đặt d=Pr(flag|A,h), f=Pr(flag|D,h): IC cần `(d−f)F >= immediate deviation gain`. Với nhiều rounds, continuation còn gồm model/link changes và future false-seizure risk; phải kiểm `G_i(h;F)−(d−f)F<=0` với G là operating reward plus đầy đủ discounted continuation. Không được thay p_min bằng d−f trong Z_T bound rồi bỏ qua future false flags hoặc participation loss. Nếu d<=f và current gain dương thì tăng F không sửa được terminal incentive.

**Corollary bảo thủ khi audit có false positives.** Giữ các funding/market assumptions, chỉ thay assumption2–3 bằng hai bounds **conditional tại mọi active history trước audit**: noncanonical/default có first-seizure probability ít nhất d_min; canonical D có false-seizure probability mỗi round không quá f_max. Cả hai là overall probabilities gồm audit selection/availability, không phải ROC conditional riêng trên audited samples.

Từ first deviation với h rounds còn lại, expected discounted loss của always-D không quá `F*f_max*Z_h`: probability first false seizure tại mỗi future round không quá f_max, và bond chỉ bị thu một lần. Deviation mất ít nhất `d_min*F` ngay round đầu; market advantage vẫn không quá Z_h. Vì vậy:

$$
\Delta\widetilde J_i(h)
\le Z_h-F_i(d_{\min}-f_{\max}Z_h).
$$

Một sufficient condition cho mọi remaining horizon là:

$$
\boxed{d_{\min}>f_{\max}Z_T,\qquad
F_i\ge\frac{Z_T+\eta}{d_{\min}-f_{\max}Z_T}.}
$$

Đây là robust continuation certificate, không chỉ thay p bằng one-round d−f. Bound rất bảo thủ, dùng history-uniform ceilings chứ không average detector accuracy. Nếu denominator không dương, **certificate này** không kết luận được, không chứng minh mọi mechanism đều bất khả thi. Với T20/gamma=.95, illustrative d_min=.25/f_max=.001 cho bond≈54.13963; f_max=.02 làm denominator âm dù one-round d>f. Honest node vẫn chịu false-exclusion/capital-loss risk, cần đưa vào participation và quality; không gọi collateral rule này “không loại nhầm” khi f_max>0. Các rates này chưa được đo trên implementation thật.

Trong một stylized stationary repeated game, nếu private loss từ bị cắt hợp tác thật sự ít nhất ell>0 mỗi round trong L rounds và detection gap là delta>0, một sufficient one-shot condition có dạng `gain <= delta*[F + gamma*ell*(1−gamma^L)/(1−gamma)]` cho gamma<1. Đây không phải theorem đã xác nhận cho DFL đang drift: phải chứng minh lower bound ell trước, và L=0 ở final action. Degree hay số token không thay cho ell. Current participation diagnostics bên dưới cho thấy ngay private cooperation surplus cũng không mặc định dương.

### 21.6. Magnitude của bond, capital và participation

T20/gamma=.95 cho Z20≈12.83028. Illustrative strict margin eta=.01, **không phải calibrated policy parameter**:

| Effective detection floor p_min | Sufficient single episode bond F |
| ---: | ---: |
| 1.00 | 12.84028 |
| 0.50 | 25.68056 |
| 0.25 | 51.36113 |

Đơn vị là một-period maximum normalized market revenue v_i*M, không phải attack cost .001. Bond là capital-at-risk, không phải honest fee51.36 mỗi round; vẫn phải kiểm liquidity, financing/verification costs và legal enforceability. Single-bond design chấm dứt admitted sending sau fault để không hứa phạt nhiều lần bằng cùng một khoản tiền đã hết.

Worst-case bound rất bảo thủ; nó **không** làm một fine lớn trở thành economically appropriate. Cần đồng thời `F_i <= available collateral_i` và **participation**:

$$J_i^{D,\mathrm{operating}}(W)-K_i-
\mathbb E_D\!\left[\sum_{t=0}^{T-1}\gamma^tF_iY_i^t\right]
\ge J_i^{outside}.$$

Operating return ở đây phải tính đúng contract dynamics, kể cả model/link changes sau false exclusion; không mặc định dùng clean-monitor trajectory khi f_max>0. Ở perfect completeness, false-seizure term bằng0 và all-D graph không bị rule này cắt. Pilot comparisons bên dưới là no-audit economic diagnostics, không phải measured IR của noisy contract.

Nếu cố định max feasible bond F_bar trước, mechanism chỉ có universal certificate khi p_min*F_bar>Z_T. Nếu không, phải tăng chất lượng audit, tìm state-dependent gain bound được chứng minh chặt hơn, hoặc công bố miền bảo đảm hẹp hơn. Maximum observed gain trong ba seeds không phải bound cho mọi state/policy. Không tăng F hoặc giả K=0 chỉ để bảng payoff chuyển sang D.

Nếu local proof generation/communication/audit overhead **còn có thể tiết kiệm bằng deviation/default**, assumption fixed K không thỏa. Phải thay numerator Z_T bằng một bound B_T bao gồm cả maximum avoidable real costs; ideal-audit condition thành `p_min*F >= B_T+eta`, noisy condition thành `(d_min−f_max*Z_T)*F >= B_T+eta`. Không được gọi actual energy/prover cost là “prepaid” chỉ để bỏ nó khỏi incentive calculation. Fixed prepaid service contract là một construction có điều kiện; measured implementation phải xác định khoản nào thật sự sunk và khoản nào còn tránh được.

Bound Z_h là tight trên lớp bounded-revenue games: một deviation có thể tạo revenue windfall1 cho mọi round còn lại, còn D nhận0; audit chỉ lấy bond và không xóa windfall. Khi đó gain chính xác `Z_h−pF`. Đây là analytical counterexample, không phải MNIST trajectory. Audit coin công khai trước send làm theorem thất bại: attack chỉ ở rounds không audit, vẫn có gain mà không mất tiền.

### 21.7. Evidence từ current game: pooling và voluntary cooperation không tự đủ

[Analysis runner](../experiments/mnist_incentive_certificate.py), [boundary tests](../experiments/test_mnist_incentive_certificate.py), [conditional certificate/diagnostics](../experiments/results/mnist_incentive_certificate.json). Chương trình không triển khai detector hoặc dùng hidden A label làm audit; nó kiểm bound toán học, rescore revenue pooling trên frozen trajectories và đối chiếu participation reference. Không retrain policies dưới utility mới, không dùng official test.

Budget-balanced pooling `r_i=(1−alpha)*sigma_i+alpha*(sum_j sigma_j)/n−c_i` vẫn có **26/144 profitable frozen unilateral deviations** trước D ngay cả alpha=1. Ví dụ binary/non-IID seed19/actor1/attack-if-send có full-pool gain≈+.11413. Vì outside option, poisoned trajectories không nhất thiết giảm total market revenue; “chia đều revenue là đủ” bị counterexample ngay trong artifacts hiện tại. Đây là finite measured-policy diagnostic, không phải solve game với agents đã thích nghi dưới pooling.

Clean monitor so với cùng-root both-local-only reference còn cho:

| Non-IID seed | ΔJ actor0, clean D/D−both-local-only | ΔJ actor1 | Total surplus |
| ---: | ---: | ---: | ---: |
| 7 | +0.24109 | -0.43364 | -0.19256 |
| 19 | -0.52860 | +0.13753 | -0.39107 |
| 31 | +0.19395 | -0.22052 | -0.02657 |

Với **những fixed clean trajectories/outside reference này**, không budget-balanced side payment nào làm cả hai node thích clean cooperation hơn outside, kể cả trước audit/financing cost. Không diễn giải thành impossibility của mọi W hoặc mọi market model. Audit quality tăng không tự đồng nghĩa private/aggregate commercial profit tăng; market và pilot-audit là hai populations/metrics khác nhau.

**Khuyến nghị có scope:** dùng audited collateral làm candidate contract cho consortium đã commit membership, vì có proof post-entry deterrence rõ và không phạt honest non-IID theo heuristic loss. Nhưng chưa gọi nó là self-sustaining voluntary DFL solution: phải có verifiable execution/settlement thật, affordable collateral, measured overhead, và clean W/kinh tế hợp tác tạo participation surplus. Nếu không chấp nhận enforcement mới, chỉ được đưa ra conditional local-monitoring IC và thực nghiệm; history/connectivity một mình không cung cấp unconditional proof mà current information chưa hỗ trợ.

