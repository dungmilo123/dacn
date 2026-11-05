# Roadmap — Chống parameter poisoning trong DFL bằng incentive và EGT

Ngày cập nhật: 2026-10-28.

**Trạng thái:** empirical episode-policy game384 cells vẫn cho opportunity-aware poisoning có lợi. Theo yêu cầu redesign reward/payoff, đã đề xuất **audit-backed collateral contract** với first-deviation deterrence proof có điều kiện, terminal settlement và participation constraints tại formulation §21. Chưa triển khai prover/escrow hoặc claim end-to-end deterrence; revenue pooling đơn thuần vẫn có26/144 profitable frozen deviations, clean non-IID market surplus hiện tại chưa đủ để cả hai tự nguyện tham gia. Official test chưa dùng; historical evidence giữ nguyên.

## 1. Câu hỏi nghiên cứu và bối cảnh đã chốt

> Trong DFL giữa các node cạnh tranh khách hàng, cơ chế aggregation weight dựa trên observation cục bộ có thể làm parameter poisoning mất lợi thế, hạn chế hành vi tấn công và duy trì quality toàn mạng hay không?

- Mỗi node có identity cố định nhưng không có phe A/D bất biến. Mỗi round chọn đúng một action: **D gửi tham số hợp lệ**, **A gửi tham số bị thao túng**.
- A vẫn có model local riêng không bị thao túng trực tiếp và dùng cùng receive-side defense như D. Không ép attacker tự phá model mình.
- Utility chung dạng hàm: **thị phần trên tổng khách hàng tiềm năng trừ cost**; tham số có thể khác giữa node nhưng không đổi mục tiêu khi đổi A/D.
- Tổng khách hàng tiềm năng cố định, có outside option. Kỳ vọng quality tăng ngoại sinh và có giới hạn; không lấy average accuracy làm ngưỡng.
- Node nhận message từ lượt trước, aggregate và nhận feedback thị trường, sau đó chọn A/D để tạo message cho lượt sau. Action cuối có bước nhận/aggregate/market settlement cuối, không bị bỏ payoff.
- Chỉ dùng thông tin cục bộ: model, validation riêng, message nhận được, share riêng và thông tin công khai về round/ngưỡng/cost. Không cấp quality thật, action kín hoặc utility của đối thủ cho policy/defense.
- Baseline chưa dùng history window/reputation. Tham số học và bộ nhớ tạm để quy feedback đúng action vẫn được giữ.
- Tách **răn đe chiến lược** khỏi **giảm damage**. All-D, utility tăng hoặc thị phần tăng không thay được bằng chứng quality.

Đặc tả toán, information boundary, giả định và ví dụ payoff: [GAME_FORMULATION.md](GAME_FORMULATION.md). [topics.md](topics.md) giữ nguyên phạm vi đề tài.

## 2. Quyết định kỹ thuật sau pilot

| Thành phần | Quyết định cho bước nghiên cứu kế tiếp |
| --- | --- |
| Task | MNIST: phân loại chữ số viết tay **0–9**, ảnh 28×28; không bỏ lớp 0 |
| Model | MLP 784→64 ReLU→10 logits, cross-entropy; chung initialization; SGD không momentum, lr=0.1, batch=64 |
| Training | Mỗi node 1,024 train samples; 1 local epoch mỗi action round |
| Dữ liệu gốc | Chỉ dùng 60,000 ảnh train của MNIST; 10,000 official test chưa download hoặc sử dụng |
| Split pool | Stratified theo label: khoảng 80% train; 20% còn lại chia thành local-validation, market và pilot-audit disjoint |
| IID | Chia từng lớp gần đều cho các node rồi lấy số sample cố định |
| Non-IID | Với mỗi lớp, lấy class-to-client proportions từ Dirichlet alpha=0.5; cùng proportions cho train/validation nhưng index rời nhau |
| Validation | 200 samples/node theo phân phối cục bộ; không cấp market/audit cho W |
| Market | 1,000 ảnh, 100/lớp từ market pool; 10 nhóm chữ số, mu_g=0.1; chỉ simulator dùng để tạo share |
| Pilot-audit | 1,000 ảnh khác, 100/lớp; chỉ đo development metrics. Không phải final test; đã dùng xem kết quả pilot |
| Mạng pilot | 4 node trên ring, nhận 2 neighbor; mỗi message cùng bản gửi tới mọi neighbor |
| Attack | Untargeted amplified sign-flip local update, scale=5; giữ bản local sạch |
| W0 | Validation-loss gate: nhận neighbor nếu loss(message) ≤ loss(local model) + **1.0 nat**; uniform weight giữa self và các neighbor được nhận |
| W1 | Game hai node: chọn weight neighbor trong {0, 0.125, 0.25, 0.5} để minimize loss validation local của model **sau aggregate**; tie chọn weight thấp hơn |
| Weight 0 | Được phép; nếu không nhận ai thì self-weight=1; phải so với local-only và clean DFL |
| Thị trường | Expected share với outside option; kappa=10; M=1,000 khách hàng tiềm năng là quy ước bối cảnh, không lấy mẫu số đếm trong pilot |
| Kỳ vọng quality | tau_t=min(0.8, 0.1+0.05t), tính ở market settlement t; thêm đối chứng tau cố định ở thí nghiệm tiếp |
| Cost chuẩn hóa | c_D=0.005; c_A=0.006. Đây là giả định kinh tế cho pilot, **không phải cost tính toán đã đo/calibrate**; phải khảo sát chênh lệch về 0 |
| Learner D | Baseline payoff-based rule mục 9.3 của formulation: p_A ban đầu 0.5, eta=0.1, bound C=0.006, không thêm exploration/mutation |
| Giới hạn learner | Chỉ dùng payoff một chuyển tiếp; chưa tối ưu long-horizon J và không chứng minh Nash. Đã có oracle best response dài hạn để đối chiếu, chưa có learner dài hạn local |
| Horizon | Pilot T=20 action rounds và 1 settlement cuối. Game oracle đã chạy horizon 1–3, gamma=0.95; gamma chưa dùng trong learner pilot |
| Seed | 7, 19, 31; 3 seed chỉ đủ pilot, không dùng để claim significance hoặc robustness rộng |

**Revision 2026-11-03 — cấu hình mạng lớn.** Bảng trên mô tả pilot 4-node ring đã chạy (alpha=0.5, pool 3 phần bằng nhau). Cấu hình hiện dùng cho lattice 4x4 thay đổi đúng bốn điểm và phải khai trong mọi run mới:

- `--topology lattice --nodes 16`: von Neumann, mỗi node 2–4 neighbor.
- `--alpha 1.0`: Dirichlet alpha=1 cho class-to-client; alpha=0.5 không khả thi ở 16 node.
- `--pool-layout balanced`: market và pilot-audit giữ đúng 100 ảnh/lớp, phần dư (~6,000 ảnh trước đây không dùng) chuyển sang local-validation, cho 10,005 ảnh. Layout `legacy` không cấp đủ 200 validation/node cho 16 client, nên `make_split` raise thay vì resample.
- `--loss-tolerance 1.0` làm default: ở non-IID alpha=1, tolerance 0.1 làm gate thấp hơn cả attack_uniform không phòng thủ.

Artifacts: [mnist_pilot_non_iid_alpha1.json](../experiments/results/mnist_pilot_non_iid_alpha1.json) (tolerance 0.1, giữ làm đối chứng âm) và [mnist_pilot_non_iid_alpha1_tol1.json](../experiments/results/mnist_pilot_non_iid_alpha1_tol1.json) (tolerance 1.0, cấu hình hiện dùng).


### Attack và candidate weight chính xác

Sau aggregate, model phục vụ là w_i^t. Local training tạo ell_i^t. Message:

- D: m_i^t=ell_i^t.
- A: m_i^t=w_i^t−5(ell_i^t−w_i^t).

Do đó norm của update gửi so với model trước training bằng 5 lần norm update local, theo hướng ngược. Không cho attacker đổi scale theo kết quả hoặc truy cập dữ liệu/quality kín của neighbor.

W0 đánh giá **toàn bộ bản message** trên validation của receiver. Nếu được nhận thì chia đều weight với self và các message khác được nhận. Rule không đọc nhãn A/D; accepted/rejected theo action thật chỉ dùng phân tích log ngoại tuyến.

**W0 là reference candidate để nghiên cứu, không phải một defense mới đã chứng minh đóng góp.** Cần dùng phân tích game để xác định hoặc sửa rule nhằm thay đổi incentive, thay vì chỉ dựa vào accuracy dưới attack.

## 3. Pilot và game nhỏ — quyết định từ bằng chứng

### 3.1. Thiết kế và artifacts

Chương trình: [experiments/mnist_pilot.py](../experiments/mnist_pilot.py). Thư viện duy nhất: [requirements-pilot.txt](../requirements-pilot.txt), NumPy 2.2.6; Python 3.12.3, BLAS mặc định 1 thread trên máy chạy pilot.

Bảy đối chứng: local-only; clean uniform; clean W0; một attacker cố định + uniform; một attacker cố định + W0; mọi node adaptive + uniform; mọi node adaptive + W0. Fixed attacker chỉ là stress baseline, không gán identity cố định cho game chính.

Artifacts giữ config, source SHA-256, checksum dữ liệu, toàn bộ index split, label counts và log từng round:

- [mnist_smoke.json](../experiments/results/mnist_smoke.json): seed 7, 8 round, model nhỏ, gate tolerance=0.1.
- [gate_tolerance_05.json](../experiments/results/gate_tolerance_05.json), [gate_tolerance_10.json](../experiments/results/gate_tolerance_10.json): calibration dung sai gate ở non-IID seed 7.
- [mnist_pilot.json](../experiments/results/mnist_pilot.json): 42 run, cấu hình chính trong bảng trên.

Calibration dùng dữ liệu development và seed 7 cũng nằm trong pilot chính; không coi pilot chính là đánh giá độc lập sau tuning. Official MNIST test chưa dùng.

### 3.2. Quality cuối episode

Q_net là mean accuracy trên tập node cố định và 10 nhóm chữ số có weight bằng nhau. Số dưới là **% accuracy, mean ± sample standard deviation qua 3 seed**, không phải confidence interval.

| Đối chứng | IID | Non-IID alpha=0.5 |
| --- | ---: | ---: |
| Local-only | 87.48 ± 0.74 | 70.99 ± 3.88 |
| Clean uniform DFL | 89.69 ± 0.74 | 87.95 ± 1.38 |
| Clean W0 | 89.69 ± 0.74 | 87.73 ± 1.50 |
| Fixed attacker + uniform | 26.93 ± 0.72 | 49.58 ± 7.23 |
| Fixed attacker + W0 | 86.14 ± 2.25 | 85.62 ± 2.62 |
| Adaptive A/D + uniform | 16.03 ± 17.22 | 9.95 ± 0.45 |
| Adaptive A/D + W0 | 84.69 ± 1.84 | 80.43 ± 3.21 |

Kết quả cho phép giữ task, attack scale=5 và candidate W0 để nghiên cứu tiếp: clean learning có lợi, attack thật sự gây damage, W0 giảm damage và không chỉ rút toàn bộ về self-only trong cấu hình đã chọn.

### 3.3. Những kết luận không được suy quá mức

- Gate tolerance=0.1 hoặc 0.5 ở smoke non-IID cho quality bằng local-only (61.95%), mất lợi ích clean DFL (78.80%). Loại cấu hình gate chặt này khỏi baseline chính, không che kết quả âm.
- Tolerance=1.0 ở calibration giữ được clean learning (79.10%) và quality dưới fixed attack (79.30%) trong smoke đó; đây là cơ sở chọn candidate, không phải optimum tổng quát.
- Trong 20 round adaptive chính, mean attack rate vẫn khoảng 0.50 ở cả uniform và W0; mean p_A cuối W0 khoảng 0.511 (IID) và 0.512 (non-IID). **Chưa có bằng chứng răn đe, hội tụ all-D hay equilibrium.**
- Trong fixed-attack W0, 50/120 message độc được nhận ở IID và 21/120 ở non-IID; message hợp lệ được nhận lần lượt 360/360 và 250/360. Không gọi gate là bộ phân loại attack hoàn hảo; some update bị thao túng có thể gây damage nhỏ/không gây damage ở state đó.
- 20 round, 4 node, một attack và MNIST chưa đủ cho robustness/real-world claim. Chưa đo tách riêng overhead aggregation, chưa có cost thực tế hoặc kiểm chứng thị trường bằng dữ liệu khách hàng thật.
- Q_net, outside share, utility và tỷ lệ A không đồng nghĩa nhau. Thị trường có thể phục vụ tốt bằng các model chuyên biệt trong khi mean quality mọi model còn thấp; tiếp tục báo theo node/nhóm và Q_min.

### 3.4. Kiểm tra đã thực hiện

- Chạy chương trình thật: tải train IDX, kiểm MD5, train, trao đổi message, aggregate, tạo market share và settle payoff.
- Kiểm tra split không trùng giữa mọi node/pool; 20 action có đủ 20 feedback và market shares cộng outside bằng 1 trong mọi round.
- Đối chiếu toàn bộ p-update trong log với công thức feedback trễ, không dùng counterfactual của A/D.
- Kiểm tra gradient MLP bằng finite differences float64, sai số tuyệt đối tối đa khoảng 1.24e-11 trên các weight/bias được chọn. Đây không phải chứng minh numerical correctness toàn bộ thư viện.

Tổng elapsed các scenario trong run chính khoảng 23.69 giây; command chính khoảng 25.5 giây ở cấu hình này, không bao gồm setup/download. Không dùng con số này để ngoại suy tài nguyên final.

### 3.5. Game hai node đã chạy

Code: [mnist_game.py](../experiments/mnist_game.py), [test_mnist_game.py](../experiments/test_mnist_game.py). Kết quả: [mnist_game.json](../experiments/results/mnist_game.json), [verification](../experiments/results/mnist_game_verification.json); model snapshots được giữ trong [mnist_game_snapshots](../experiments/results/mnist_game_snapshots/).

Tái lập bitwise cần cùng NumPy/BLAS và số thread. CLI import pilot để đặt BLAS mặc định 1 thread **trước** khi import NumPy. Khi replay trong một process khác, đặt `OPENBLAS_NUM_THREADS=1` trước import NumPy; import NumPy rồi mới đặt biến môi trường có thể cho model hash khác dù cùng seed. Đã replay từ NPZ qua hai transition AD→DA và khớp model/weight chính xác với cấu hình đúng.


- Native two-node split, giữ model/task/attack/W0 từ pilot; không lấy hai node đã train trong ring 4 node rồi coi là game hai node.
- 3 seed × IID/non-IID × snapshot sau 0/5/10 clean-uniform rounds = **18 snapshot**. Từ mỗi snapshot so uniform, W0 và self-only trên đúng cùng model, dữ liệu và minibatch schedule.
- Lưu 18 model snapshots bằng NPZ và SHA-256. Mỗi cây có 85 state, 84 transition, depth=3; tổng **54 cây, 4,590 state, 4,536 transition**.
- Liệt kê đầy đủ mọi action profile tại mọi state; reward dùng market, không dùng pilot-audit. Audit chỉ đo Q_net/Q_min.
- Phân tích 8 scenario/cây: cost-gap 0, 0.001, 0.01, 0.05 × tau scheduled hoặc cố định 0.8; tổng **432 stage game**. Cost/tau được replay trên cùng model tree vì uniform/W0/self hiện tại không đọc share/cost; không áp dụng cách reuse này nếu W tương lai phụ thuộc chúng.

**Phạm vi exact:** solver biết toàn state và transition với minibatch order được cố định theo seed. Đây là deterministic seeded oracle game, không phải expected payoff đã tích phân mọi ngẫu nhiên SGD, không phải policy cục bộ triển khai được. Trong trường hợp đối thủ iid_half, solver lấy kỳ vọng đúng trên các action coin của đối thủ và chọn trước coin hiện tại.

### 3.6. Stage Nash và khả năng A xâm nhập D

Cấu hình gốc: cost-gap=0.001, tau scheduled. Bảng đếm trên 18 snapshot có tương quan qua warmup, không phải 18 quan sát độc lập hoặc ước lượng xác suất equilibrium ngoài thực tế.

| Cơ chế | Nash AA | Nash AD/DA | Nash DD | All-D chống được A hiếm, nghiêm |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 18 | 0 | 0 | 0 |
| W0 | 12 | 3 | 3 | 3 |
| Self-only | 0 | 0 | 18 | 18 |

Ba DD của W0 đều ở cold-start non-IID, seed 7/19/31. Trong cả bốn profile, weight là identity: W0 từ chối **mọi neighbor, kể cả D**. Quality sau một bước lần lượt 48.4%, 41.0%, 49.6%, không phụ thuộc action. Vì vậy đây là trạng thái self-only cộng chi phí attack, **không phải bằng chứng răn đe mà vẫn giữ lợi ích hợp tác**.

Ví dụ IID seed 7, snapshot sau 5 warmup rounds, W0 nhận cả hai chiều ở mọi profile:

| Profile | Payoff node 0 | Payoff node 1 | Q_net sau một bước |
| --- | ---: | ---: | ---: |
| DD | 0.49126 | 0.49126 | 84.90% |
| AD | 0.60540 | 0.37579 | 79.40% |
| DA | 0.36199 | 0.61832 | 79.00% |
| AA | 0.42674 | 0.52788 | 73.50% |

A có lợi khi đơn phương lệch khỏi DD ở cả hai node: Delta u lần lượt khoảng 0.11414 và 0.12706. Gate không chặn những message độc này ở state đó; việc giảm damage trong pilot 4 node không tạo ra bảo đảm incentive.

Với self-only/cost-gap=0, mọi action mixture đều indifferent; solver giữ continuum thay vì ép DD. Có scenario W0 non-IID seed 7/snapshot 5/cost-gap=0.05/tau scheduled chỉ có mixed Nash: p0(A)≈0.32144, p1(A)≈0.86214. Không chọn cost cao này làm cơ chế răn đe mới; nó chỉ kiểm tra sensitivity và mixed solver.

Replicator được phân tích theo hai xác suất p_i, tính Jacobian/eigenvalues trong game đóng băng và chạy ODE minh họa. Không symmetrize payoff hoặc rút thành một tỷ lệ x, không gọi đó là ESS của DFL động. Khi eigenvalue có real part bằng 0, chỉ báo nonhyperbolic/không kết luận ổn định; stationary continua ở trường hợp suy biến chưa được phân loại đầy đủ.

### 3.7. Best response 1–3 bước và kiểm chứng

Đối thủ cố định: always-D, always-A, alternating D/A và iid_half. Solver chọn trước action đối thủ hiện tại; tương lai có thể thích nghi theo state full-information. Gamma=0.95, tie-break chọn D nhưng giữ cả tập best actions trong log. Đây là best response với đối thủ cố định, không phải dynamic Nash.

Ở horizon 3, mỗi cơ chế có 18 snapshot × 2 player × 4 policy đối thủ = 144 bài toán:

| Cơ chế | Root myopic A → dài hạn D | Root myopic D → dài hạn A | Tổng đảo lựa chọn nghiêm |
| --- | ---: | ---: | ---: |
| Uniform | 38 | 0 | 38/144 |
| W0 | 5 | 10 | 15/144 |
| Self-only | 0 | 0 | 0/144 |

Ví dụ W0, IID seed 7/cold-start, node 1 trước always-D: immediate payoff D≈0.48937 > A≈0.46837, nhưng oracle Q ba bước D≈1.43217 < A≈1.43650. Hậu quả dài hạn thực sự có thể đổi quyết định; không được dùng learner một bước làm bằng chứng tối ưu J.

Kiểm chứng đã chạy:

- 5 regression tests: matching pennies cần mixed strategy; profitable defection/invasion; equilibrium continuum; không nhìn current opponent coin; quyết định dài hạn đảo myopic.
- Script độc lập liệt kê **128 deterministic contingent policies** theo lịch sử action đối thủ cho horizon 3, khớp **432 best responses** cấu hình gốc. Lịch sử này chỉ thuộc miền oracle kiểm chứng, không được cấp cho policy local.
- Kiểm expected regret bằng 0 tại representative của **802 equilibrium cells** trong 432 stage scenario; kiểm zero-cost continuum, weight simplex, tree depth và model/source hashes.
- Run chính khoảng 44.1 giây elapsed trong chương trình. Đây vẫn là development evidence; official test chưa được download hoặc sử dụng.

**Kết luận hành động:** giữ W0 làm reference baseline giảm damage, không coi nó là cơ chế incentive đã thành công. Thiết kế tiếp phải kiểm tra đồng thời A-invasion/Delta Q, quality và việc giữ update hợp lệ; không chấp nhận nghiệm self-only chỉ vì có DD.

### 3.8. W1 — validation trên mixture và kết quả âm

**Giả thuyết:** W0 có thể nhận một message làm aggregate xấu đi vì tolerance 1.0 và weight cố định. W1 chọn weight bằng loss của chính mixture được phục vụ, có self-only trong tập ứng viên để không bắt buộc nhận update độc. Rule/formula và information boundary ở [formulation §12](GAME_FORMULATION.md#12-w1-chọn-mixture-bằng-validation-local-và-giới-hạn-incentive).

Run [mnist_w1_game.json](../experiments/results/mnist_w1_game.json): 18 snapshot gốc × 4 cơ chế, horizon 3, cùng 8 biến thể cost/tau. Grid {0, 0.125, 0.25, 0.5} chốt trước run; không chọn lại bằng audit. Giữ nguyên attack, utility, cost và bootstrap. Đây là W1 **hai node**, không phải kết quả pilot ring bốn node. Tổng 72 cây/6,120 state/6,048 transition; elapsed trong chương trình khoảng 63.9 giây.

Ở cost-gap=0.001, tau scheduled:

| Cơ chế | Nash AA | Nash AD/DA | Nash DD | DD có chia sẻ ở root | Node-state A có lợi khi lệch khỏi DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 18 | 0 | 0 | 0 | 36/36 |
| W0 | 12 | 3 | 3 | 0 | 26/36 |
| W1 | 5 | 7 | 6 | 0 | 16/36 |
| Self-only | 0 | 0 | 18 | 0 | 0/36 |

Các snapshot tương quan theo seed/warmup; đây không phải xác suất equilibrium hay significance test. Cột DD có chia sẻ đếm snapshot vừa strictly resist A ở cả hai node vừa có ít nhất một neighbor weight dương khi DD. Cả 6 DD của W1 đều có weight identity khi nhận D; không tính đó là thành công.

W1 loại **36/36 message A** trong unilateral root deviations, trong khi W0 loại 24/36. Mean stage A-invasion gain qua 36 node-state giảm từ 0.04588 (W0) xuống 0.00650 (W1), nhưng dấu vẫn dương ở 16 trường hợp. Khi mở toàn bộ cây W1, **156/1,512 message A** vẫn có weight dương; local-validation selection không phải bảo đảm chặn poison.

**Vấn đề incentive không chỉ là lọc:** nếu D giúp đối thủ cải thiện nhưng A bị loại, chọn A vẫn có thể tước lợi ích chia sẻ của đối thủ mà giữ model riêng. Ví dụ IID/seed 7/snapshot 5/W1, node 0:

- DD: own market accuracy 84.2%, neighbor 84.3%; payoff node 0 = 0.488682.
- AD: own market accuracy vẫn 84.2%, neighbor 84.2%; message A weight=0 tại receiver, payoff node 0 = 0.489500.
- Gain A≈0.000819 sau extra cost 0.001; receiver không bị kéo dưới model local vì A bị loại, nhưng mất ích lợi của D. Formula theo từng nhóm trong formulation giải thích hiệu ứng này; average accuracy không tự quyết định dấu payoff.

### 3.9. Quality, horizon và quyết định sau W1

Quality audit sau **ba transition** all-D/all-A từ mỗi snapshot; mỗi ô là mean qua 3 seed × 3 snapshot của cùng mode. Không phải final training accuracy hoặc estimate từ 9 run độc lập. All-A là stress path, không ép learner/equilibrium thực sự chọn path đó.

| Cơ chế | IID all-D Q_net (%) | IID all-A Q_net (%) | Non-IID all-D Q_net (%) | Non-IID all-A Q_net (%) |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 85.92 | 14.62 | 79.93 | 13.43 |
| W0 | 85.92 | 83.27 | 72.76 | 66.07 |
| W1 | 85.95 | 85.12 | 68.54 | 67.87 |
| Self-only | 85.31 | 85.31 | 67.72 | 67.72 |

W1 giữ clean IID gần W0, nhưng clean non-IID giảm **4.22 điểm phần trăm** so W0 và chỉ hơn local-only khoảng 0.82 điểm. Loss validation local không đại diện quality cân bằng theo nhóm của toàn thị trường. Không chọn W1 làm defense cuối chỉ vì AA ít hơn hoặc attack damage giảm.

Ở horizon 3/W1 có 9 root myopic A→oracle D và 2 D→A trong 144 response problems. So nguyên sequence all-D với oracle best response trước always-D: W1 có 5/18 snapshot không có unilateral deviation cải thiện return, nhưng cả 5 đều self-only tại root. Chưa thấy cooperation hữu ích được giữ bằng hiệu ứng dài hạn trong miền đã chạy. Phép so rooted/seeded này không phải subgame-perfect/dynamic Nash hoặc định lý cho mọi horizon.

Kiểm chứng mới trong [mnist_w1_verification.json](../experiments/results/mnist_w1_verification.json):

- 7 regression tests, gồm mixture hữu ích dù full message xấu và không ép chia sẻ với message có hại/identical.
- 54 cây baseline có toàn bộ node/edge/analysis **giống hệt** artifact reference trước khi thêm W1.
- 3,024 receiver-selection checks W1: chọn đúng grid minimum, loss không hơn self trên validation dùng chọn, weight simplex.
- Tái tính payoff từ market quality; kiểm regret tại 1,058 Nash cells; kiểm 6,048 counterfactual own-quality invariants trên cả bốn cơ chế.
- Liệt kê độc lập 128 contingent policies, khớp 144 best responses horizon 3 của W1.
- Kiểm 18 snapshot digests; retrain/replay 4 transition W1 trên hai nhánh, gồm một state sâu nhận message A, khớp model hashes, weight, quality và mọi grid loss.

**Quyết định sau W1:** giữ W1 làm kết quả âm/reference candidate, không tăng cost, đổi utility hoặc ép self-only để báo thành công. Đã nghiên cứu và chạy extension reciprocity có history riêng ở mục 3.10; baseline memoryless không đổi.

### 3.10. Reputation–reciprocity: thực thi được access, chưa đủ incentive

Code [mnist_reciprocity.py](../experiments/mnist_reciprocity.py), [raw results](../experiments/results/mnist_reciprocity.json), [verification/tổng hợp paired outcomes](../experiments/results/mnist_reciprocity_verification.json). Protocol đầy đủ, information boundary và điều kiện incentive ở [formulation §13](GAME_FORMULATION.md#13-extension-reputationreciprocity-quyền-nhận-có-điều-kiện).

**Bối cảnh mở rộng đã chốt cho run:** reputation trực tiếp theo từng cặp, không global score/server; quyền gửi song phương có thể từ chối recipient. Stable identities, không public-broadcast/relay để bypass, A chỉ thao túng params và vẫn tuân protocol. Đây là giả định mạnh hơn receive-side weight thuần túy, phải giữ trong claim.

Evaluator chấm 1 nếu mixture tốt nhất trên grid W1 cải thiện validation local hơn 1e-6 nat, còn lại 0; EMA beta=0.5. Sender chỉ gửi cho recipient có reputation do chính sender chấm ≥0.75; probe công khai mỗi 2 round để phục hồi. Không nhận message thì giữ điểm, không phạt thiếu bằng chứng. Permissions từ state cũ, score hiện tại chỉ tác động lượt sau. Receiver vẫn dùng W0 trên message được giao.

Hai nhánh: `reputation_only` chấm nhưng không enforcement và `reciprocal` có enforcement. Cùng 18 model snapshot 0/5/10, R khởi tạo 1 ở mỗi root, không coi đó là history warmup đã học. Giữ nguyên utility/cost, không cộng tiền phạt hoặc tiết kiệm communication cost giả định.

- 36 cây/3,060 state/3,024 transition, exact oracle best response H1–3 trước bốn opponent policies cũ; 8 biến thể cost/tau.
- 180 rollout × 20 transition: all-D, node 0 A một lần rồi D, node 0 always-A, node 0 alternating A/D, cả hai iid_half. Các action sequence giống nhau giữa monitor/enforcement; không phải learned equilibrium.
- Elapsed trong chương trình khoảng 94.7 giây; không phải benchmark overhead tách riêng.

**Clean exclusion:** mean final audit Q_net sau 20 transition, mỗi ô qua 3 seed × 3 snapshot/mode:

| Mode, cả hai always-D | Monitor Q_net (%) | Reciprocal Q_net (%) | Directed deliveries bị chặn |
| --- | ---: | ---: | ---: |
| IID | 89.23 | 89.20 | 7/360 |
| Non-IID | 77.19 | 75.10 | 133/360 |

Trong clean non-IID, chỉ 53/227 message D được giao nhận score “useful”; IID là 341/353. Không gọi score này là xác suất attacker: nó đo usefulness trên validation lệch của receiver và có thể trừng phạt D hợp lệ. Các snapshot tương quan; không diễn giải thành kết luận thống kê dân số.

**Incentive phải so A với D trong từng mechanism.** Bảng dưới là gain discounted J của node 0 so với chính nó always-D, khi node 1 always-D, gamma=0.95, 20 transition:

| Mode | Policy deviation node 0 | Mean gain monitor | Mean gain reciprocal | Gain dương monitor → reciprocal |
| --- | --- | ---: | ---: | ---: |
| IID | A một lần rồi D | 0.03053 | 0.01052 | 8/9 → 5/9 |
| Non-IID | A một lần rồi D | 0.05566 | 0.05043 | 6/9 → 5/9 |
| IID | Always-A | 0.46961 | 0.55631 | 9/9 → 9/9 |
| Non-IID | Always-A | 0.19933 | 0.30845 | 7/9 → 7/9 |

Sanction làm một lần A bớt lợi trung bình, nhưng không loại A; với always-A, mean lợi thế tương đối còn tăng. Không suy “attacker bị chặn nên bị thiệt”: cùng always-A policy, mean Q_0 IID tăng từ 84.20% lên 87.23% khi enforcement, final share từ 40.73% lên 43.33%. Thay đổi tương tác/feedback có thể có lợi cho attacker. Non-IID có Q_0/share giảm trung bình nhưng honest baseline cũng chịu phạt, nên gain A so D vẫn tăng.

Oracle H3 trước always-D cho root strictly chọn A: IID 17/18 → 12/18, non-IID 11/18 → 10/18 player-state. Trên toàn bộ bốn opponent policies, enforcement đổi 7 root A→D nhưng cũng đổi 2 D→A trong 144 bài toán. Đây là best response với opponent cố định và seeded full state, không phải dynamic Nash hoặc khả năng tối ưu của local agent.

144 root stage payoff matrices qua cost/tau giống hệt giữa hai nhánh: reputation lạc quan cho phép mọi gửi ở root, sanction chỉ ảnh hưởng tương lai. Frozen stage replicator không thể tự chứng minh hiệu ứng này; phải dùng continuation/trajectory với state reputation.

Kiểm chứng:

- 9 regression tests, gồm mất quyền nhận đúng chiều/trễ, phục hồi qua probe, và D hợp lệ nhưng không useful bị giảm điểm.
- Monitor có 1,512 transition model/quality/weight cùng 18 full analyses giống hệt W0 reference.
- 13,248 receiver access/score checks qua tree và rollout; 756 permission checks không nhìn current action; 1,865 undelivered cases không bị tạo score giả.
- 288 H3 oracle responses khớp enumeration độc lập 128 contingent policies; 3,600 market/payoff settlement checks và 90 same-action rollout pairs.
- Retrain/replay 6 transition, gồm honest non-IID bị withholding và probe, khớp model hashes/quality/reputation/evidence; official test chưa dùng.

**Kết luận:** reputation + conditional access là cơ chế hợp lệ để tạo hậu quả trễ, nhưng candidate hiện tại chưa đạt deterrence/quality. [CFFL (2020)](https://arxiv.org/abs/2008.12161) đã dùng reputation để phân bổ update ở round sau trong FL có server; thêm reputation không tự là novelty. Contribution cần nằm ở điều kiện incentive và cơ chế hoạt động được dưới cạnh tranh/non-IID, không ở tên gọi.

**Sau run này:** đã triển khai và chạy đối chứng validation service ở mục 3.11, không dùng market/audit/official test cho scoring. Threshold/probe sensitivity và endgame vẫn chưa được khóa; không ép accuracy tuyệt đối giảm hoặc thay reward để làm reputation thành công.

### 3.11. Validation service: sửa clean exclusion, chưa sửa hết incentive

[Raw results](../experiments/results/mnist_service_validation.json), [verification/tổng hợp](../experiments/results/mnist_service_validation_verification.json), [protocol và official-test boundary](GAME_FORMULATION.md#14-validation-đại-diện-service-và-vai-trò-official-test).

Thiết kế factorial: 18 snapshot × train-matched/service validation × monitor/reciprocal access. Tổng **72 cây/6,120 state/6,048 transition**, 360 rollout × 20 transition. Elapsed trong chương trình khoảng 217.5 giây. Không đổi attack, cost, beta=0.5, threshold=0.75, probe period=2, grid hoặc action RNG.

Service validation có **200 samples/node, 20/lớp**, lấy từ phần chưa dùng của reserved local-validation population thuộc official MNIST train 60,000; disjoint giữa node và với validation cũ. Train, market, pilot-audit và root models không đổi. Đây là assumption có private labeled validation đại diện tệp service uniform, không phải cho node dùng test hoặc biết quality thật của đối thủ.

Cả W0 gate và reputation score dùng validation của nhánh. Cải thiện giữa validation modes có thể đến từ cả hai; chỉ so monitor/reciprocal **trong từng mode** mới tách effect của access enforcement. Thay validation cũng thay sample indices, nên cần thêm seed/repeated sampling trước claim nhân quả/thống kê rộng về distribution.

Clean all-D, mean audit Q_net cuối 20 transition qua 3 seed × 3 snapshot/mode:

| Train mode | Validation | Monitor Q_net (%) | Reciprocal Q_net (%) | Directed deliveries bị chặn |
| --- | --- | ---: | ---: | ---: |
| IID | Train-matched | 89.23 | 89.20 | 7/360 |
| IID | Service | 89.23 | 89.22 | 7/360 |
| Non-IID | Train-matched | 77.19 | 75.10 | 133/360 |
| Non-IID | Service | 87.13 | 87.13 | 0/360 |

Service/non-IID chấm useful cho 360/360 message D đã giao trong clean path. Quan sát này khắc phục failure cụ thể ở run cũ, không chứng minh không bao giờ loại nhầm D. Monitor phục hồi mạnh cho thấy validation đại diện service cũng sửa filtering; không gọi tăng 12.03 điểm ở reciprocal là đóng góp riêng của reputation.

Service validation, node 1 always-D; gain J của node 0 so chính nó always-D, gamma=0.95, 20 transition:

| Train mode | Deviation node 0 | Mean gain monitor | Mean gain reciprocal | Gain dương monitor → reciprocal |
| --- | --- | ---: | ---: | ---: |
| IID | A một lần rồi D | 0.03053 | 0.00994 | 8/9 → 5/9 |
| Non-IID | A một lần rồi D | 0.08165 | 0.00572 | 9/9 → 5/9 |
| IID | Always-A | 0.45276 | 0.51426 | 9/9 → 9/9 |
| Non-IID | Always-A | 0.28175 | 0.33307 | 6/9 → 8/9 |

Như vậy validation tốt hơn và sanction trễ vẫn chưa đủ bảo đảm răn đe A kéo dài. Fixed-policy comparisons không phải oracle tối ưu 20 bước, dynamic Nash hoặc thống kê từ 9 independent runs.

**Credit mismatch cần sửa tiếp:** reputation thưởng theo mixture tốt nhất trong grid, nhưng W0 thực sự dùng weight 0 hoặc 0.5. Trong service/non-IID/node0-always-A, có 70/180 delivered A được credit useful; 30 trong 70 không cải thiện validation của aggregate thực tế. Không phạt theo nhãn A thật: 40 còn lại cải thiện criterion validation thực sự, nên phải đánh giá damage/incentive riêng.

Kiểm chứng:

- 11 regression tests, gồm balanced private validation disjoint/đúng budget và fail khi reserved pool thiếu thay vì lấy lấn evaluation data.
- 12 historical split manifests 2/4 node không đổi sau extraction `partition_pools`.
- Train-matched có 36 cây và 180 rollout giống hệt run reputation lịch sử; giữ nguyên artifact/checksum cũ.
- Rebuild độc lập 18 W0 model trees với service validation, khớp model/quality/weights monitor; kiểm sáu service manifests giữ train/market/audit và actual labels/counts.
- 576 H3 best responses khớp enumeration độc lập 128 contingent policies; 26,496 receiver access/score/weight checks, 7,200 market settlement checks, 180 same-action monitor/enforcement pairs.
- Retrain/replay sáu service transition, khớp model hashes/quality/reputation/evidence và manifests; loader vẫn chỉ đọc hai official-train IDX files.

**Quyết định sau validation:** giữ service validation làm assumption minh bạch và train-matched làm sensitivity. Đã triển khai hai credit/aggregation rules nhất quán ở mục 3.13; chưa chọn final defense và không dùng official test để quyết định cách sửa.

### 3.12. Official test dùng ở đâu?

**Nên dùng cho final evaluation, chưa dùng để tiếp tục thiết kế cơ chế.** Scoring reputation, weight selection, market feedback hoặc chọn tham số từ official-test scores đều biến nó thành development data, dù node không được nhìn test trực tiếp.

Giữ official MNIST test 10,000 ảnh chưa dùng cho tới khi khóa protocol/candidate/seed/metrics. Final evaluation dùng đủ test set, báo micro accuracy và per-class accuracy; Q_net là macro qua nhóm/node, không đồng nhất với micro khi class counts khác nhau. Nếu chủ động mở test để đánh giá exploratory bây giờ rồi sửa theo kết quả, phải bỏ claim đó là final holdout độc lập và có holdout khác chưa dùng.

### 3.13. Credit thực tế: hết mismatch nhưng chưa đủ răn đe

[Protocol ở formulation §15](GAME_FORMULATION.md#15-credit-theo-aggregate-thực-tế-hai-đối-chứng-tách-biệt), [raw run](../experiments/results/mnist_realized_credit.json), [verification/tổng hợp](../experiments/results/mnist_realized_credit_verification.json). Hai rule mới:

- `realized_gate`: giữ W0, credit theo loss improvement của chính aggregate W0 được phục vụ.
- `realized_mix`: dùng W1 chọn mixture, credit theo loss improvement của chính mixture được chọn.

`legacy_gate` giữ làm reference: W0 + best-hypothetical-mixture credit, không đổi nhãn artifact cũ. Run chính có 18 root × 2 validation modes × 2 rules × monitor/reciprocal = **144 cây/12,240 state/12,096 transition**, cùng **720 rollout × 20 transition**. Same data, initial models, actions/RNG, cost, attack, beta, threshold, probe và gamma. Elapsed trong chương trình khoảng 514.0 giây; legacy replay subset 8 cây/40 rollout khoảng 19.8 giây, không phải benchmark runtime giữa rules.

**Credit invariant:** 46,582 delivered receiver cases không credit một hypothetical mixture thay cho aggregate thực tế. Điều này sửa consistency, không bảo đảm market quality hoặc strategy deterrence. Khi W0 nhận aggregate làm tăng loss, `realized_gate` hạ credit nhưng không hoàn tác damage đó.

Service validation, clean all-D, mean final audit Q_net sau 20 transition:

| Rule + reciprocity | IID Q_net (%) | Non-IID Q_net (%) | IID deliveries bị chặn | Non-IID deliveries bị chặn |
| --- | ---: | ---: | ---: | ---: |
| Legacy gate | 89.22 | 87.13 | 7/360 | 0/360 |
| Realized gate | 89.27 | 85.60 | 29/360 | 26/360 |
| Realized mix | 89.31 | 87.54 | 7/360 | 0/360 |

Với realized-mix/service/non-IID, mọi clean delivered D đều có positive realized validation improvement, nên không phải DD/self-only giả. Các ô vẫn là mean qua 3 seed × 3 snapshot tương quan, không phải 9 independent repetitions hoặc final-test scores.

Persistent deviation: node 0 always-A so chính nó always-D, node 1 always-D, gamma=0.95, 20 transition, service validation:

| Train mode | Rule + reciprocity | Mean gain J của A so D | Snapshot gain dương |
| --- | --- | ---: | ---: |
| IID | Legacy gate | 0.51426 | 9/9 |
| IID | Realized gate | 0.45039 | 9/9 |
| IID | Realized mix | 0.06188 | 7/9 |
| Non-IID | Legacy gate | 0.33307 | 8/9 |
| Non-IID | Realized gate | 0.52817 | 8/9 |
| Non-IID | Realized mix | 0.29280 | 7/9 |

Realized mix giảm mean persistent-A gain ở service cases nhưng chưa làm nó mất lợi thế. Không thắng mọi deviation: A một lần rồi D có mean gain realized-mix/reciprocal IID≈−0.01103 nhưng non-IID≈0.03864; vẫn dương ở 3/9 IID và 6/9 non-IID. Không chọn một mean âm để claim equilibrium.

H3 full-state oracle kiểm **cả hai player** trước always-D: realized-mix/service vẫn strictly chọn A tại root ở 9/18 IID và 12/18 non-IID player-state. Không có service root nào mà toàn sequence all-D không có profitable H3 deviation ở cả hai node. Node-0 fixed-policy stress test không đại diện tất cả node, policy hoặc horizon.

Kết quả train-matched/non-IID/realized-mix có node-0 fixed A không lợi ngay cả ở monitor, trong khi clean Q_net reciprocal chỉ khoảng 72.58% và có 130/360 clean deliveries bị chặn. Đây không phải bằng chứng reputation răn đe; vẫn có H3 profitable root A ở player khác/state khác.

**Giới hạn access rõ hơn:** probe vô điều kiện mỗi 2 round đảm bảo nhận ít nhất 10 message/20 transition bất kể reputation. Realized-mix/service/node0-always-A vẫn nhận 93/180 message D qua 9 root/mode, trong đó 90 là probe. Receipt không đồng nghĩa mọi message được aggregate; bound này cũng chưa chứng minh probe là nguyên nhân duy nhất của gain A.

Kiểm chứng:

- 13 regression tests; thêm hai consumer-visible cases: W0 overshoot không được credit, mixture được chọn chính là model được credit/phục vụ.
- 52,992 receiver timing/reputation/weight checks; 46,582 realized-credit checks; 23,712 actual-grid-minimum checks; 14,400 market settlement checks; 360 same-action monitor/access pairs.
- Legacy replay giữ mọi field lịch sử ở 8 cây và 40 rollout; credit-only W0 monitor giữ 36 model trees/analyses và 180 model/payoff rollouts; W1 train-matched monitor giữ 18 model trees/analyses.
- 576 W0 credit-only root stage matrices không đổi; 1,152 H3 best responses khớp independent enumeration 128 contingent policies. Stage game giống nhau không có nghĩa continuation game giống nhau.
- Retrain 24 transitions seed 7/snapshot 5 dưới cả hai validation/rule: trực tiếp tính loss model thật/grid argmin, khớp credit, model hashes và reputation của main run. 12 split manifests không đổi; source hashes kiểm khớp; official test không dùng.

**Quyết định sau credit alignment:** giữ realized-mix/service làm candidate, không coi là final defense. Access/recovery/endgame đã được kiểm ở mục 3.14; không tăng cost, bỏ recovery hoặc chọn root dễ để ép D.

### 3.14. Probe 1/2/4/8, recovery hai phía và finite endgame

[Protocol formulation §16](GAME_FORMULATION.md#16-accessprobe-sweep-với-credit-và-aggregation-cố-định), code [mnist_access_sweep.py](../experiments/mnist_access_sweep.py), [raw results](../experiments/results/mnist_access_sweep.json), [verification/tổng hợp](../experiments/results/mnist_access_sweep_verification.json).

Giữ realized-mix/service và toàn bộ credit/cost/attack/EMA/threshold/gamma/data. Chỉ sweep K=1/2/4/8; train-matched không rerun ở bước này, evidence failure cũ giữ nguyên. Probe theo absolute round, không reset phase tại snapshot. Mỗi root chạy một monitor và bốn reciprocal conditions: **90 cây/7,650 state/7,560 transition**, **900 rollout × 20 transition**, elapsed trong chương trình khoảng 373.1 giây.

Mười schedules: all-D, iid_half; một A rồi D, always-A, ba A rồi D, last-A, mỗi kiểu riêng actor0/actor1 trước peer always-D. Đây là fixed stress policies, không learned roles hoặc optimal20. So gain J với all-D trong cùng condition.

**Non-IID, mean persistent-A gain J qua 9 root/actor, gamma=0.95, 20 transition:**

| Probe K | Clean Q_net (%) | Actor 0 gain | Actor 0 gain dương | Actor 1 gain | Actor 1 gain dương |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 87.54 | 0.3950 | 6/9 | 0.5950 | 9/9 |
| 2 | 87.54 | 0.2928 | 7/9 | 0.4634 | 9/9 |
| 4 | 87.54 | 0.1299 | 4/9 | 0.3409 | 8/9 |
| 8 | 87.54 | 0.1189 | 4/9 | 0.3325 | 8/9 |

Clean non-IID có 0/360 directed deliveries bị chặn ở mọi K. Mean gain giảm không đồng nghĩa số profitable roots giảm đơn điệu, hoặc không còn profitable policy. Các roots cùng seed/warmup có tương quan; không gọi 9 root là independent repetitions.

**IID, cùng điều kiện:**

| Probe K | Clean Q_net (%) | Clean deliveries bị chặn | Actor 0 always-A gain J | Actor 1 always-A gain J |
| --- | ---: | ---: | ---: | ---: |
| 1 | 89.28 | 0/360 | 0.08893 | 0.16568 |
| 2 | 89.31 | 7/360 | 0.06188 | 0.13856 |
| 4 | 89.31 | 8/360 | 0.02700 | 0.11797 |
| 8 | 89.32 | 10/360 | 0.00373 | 0.08200 |

K=8 IID vẫn có positive always-A gain ở 4/9 actor0 và 7/9 actor1 roots. Không chọn một mean gần 0 hoặc clean Q tốt làm bằng chứng strategic stability. H3 cũng không monotone mọi metric: strict root A trước always-D tăng K4→K8 từ 7→8/18 IID và 10→11/18 non-IID player-state.

**Recovery sau ba A rồi D:** trong 144 reciprocal actor/root/K cases, 140 restored threshold eligibility, 4 eligible ngay khi quay lại D, không censored trong window burst20. IID thường cần 2 D transitions, một case K8/actor0 cần 6; non-IID conditional mean khoảng 1–2, không tăng đơn điệu theo K. K>1 đều quan sát được non-probe receipt sau eligibility ở các burst cases; K1 không có non-probe round nên metric đó unobservable, không phải recovery failure.

Clean IID có 8 low-eligibility episodes/K, 7 restored và 1 right-censored; K1 không chặn delivery dù score thấp. Không lẫn threshold crossing với packet denial hoặc gọi terminal-censored episode là mất access vĩnh viễn.

**Counterexample binary credit:** [formulation §16.5](GAME_FORMULATION.md#165-counterexample-không-chỉ-do-probe-binary-credit-bỏ-qua-độ-lớn-đóng-góp) được retrain trực tiếp. Non-IID seed31/snapshot5/K8, sau DA, actor1 có R_0,1=0.5. Gửi D cải thiện actual receiver loss≈0.49791, gửi A chỉ≈0.03259, nhưng cùng score1 và cùng R_0,1=0.75. A tăng own payoff≈0.04576, giảm Q_net 81.85%→76.95% so D, vẫn lấy lại quyền nhận ở non-probe round kế tiếp. Remaining-two-step oracle gain A≈0.04959. Counterfactual D chỉ dùng offline; không cấp nó cho scorer runtime.

Trong K8/non-IID trees có 69 lower-contribution A counterfactuals vẫn profitable với cùng next reputation, 65 trong đó giảm Q_net so D; phần lớn ở depth2 và chỉ một ở depth1. Không diễn giải số correlated tree states này thành tỷ lệ ngoài đời hoặc evidence mọi A đều harmful.

**Finite endgame:** với non-IID, last-A profitable ở cả 9 roots và cả hai actor cho mọi K; mean last-settlement gain actor0≈0.02450, actor1≈0.04539. Prefix all-D và own market quality cuối khớp; reputation đổi ở terminal không tạo continuation penalty. Các K cùng clean history nên không phải 72 independent discoveries. Không thể lấy hình phạt access sau horizon để claim all-D là best response trong game đang chạy.

Kiểm chứng:

- 15 regression tests, thêm terminal eligibility không bị tính thành receipt tương lai và recovery chưa xảy ra được right-censor.
- 18 K1 trees/180 rollouts khớp monitor; 36 historical monitor/K2 trees và 144 common-policy rollouts khớp reference; sáu service split manifests không đổi.
- 51,120 directed access/reputation checks, 45,411 actual mixture/credit checks, 18,000 market settlements, 360 recovery/censoring checks, 180 terminal prefix/own-quality checks, 720 same-action monitor/period pairs.
- 720 H3 best responses khớp enumeration độc lập 128 contingent policies; 576 reciprocal root stage matrices khớp monitor. First-step game giống nhau không chứng minh continuation game giống nhau.
- Retrain bốn real-MNIST rollouts cho smoke; ba trace (mirrored recovery/endgame) khớp main hashes/reputation. Hai counterfactual model của binary-credit example được replay riêng, khớp loss/quality/payoff/reputation.

**Quyết định sau access sweep:** chưa có K đạt full deterrence trong miền deviations đã chạy. Magnitude-sensitive ledger đã được thử ở mục 3.15, không tăng cost hoặc thêm fake terminal penalty. Learner dài hạn/policy game vẫn chưa chốt; official test vẫn giữ cho final evaluation.

### 3.15. Magnitude ledger: phân biệt credit nhưng chưa cải thiện incentive

[Protocol formulation §17](GAME_FORMULATION.md#17-earned-access-ledger-theo-mức-đóng-góp), [code](../experiments/mnist_earned_access.py), [raw results](../experiments/results/mnist_earned_access.json), [verification/tổng hợp](../experiments/results/mnist_earned_access_verification.json).

Hai ledger cùng price1, initial balance1, capacity4; mỗi packet được giao tiêu phần balance có sẵn, kể cả probe, không debt. Binary kiếm1 khi useful; magnitude kiếm fractional realized loss improvement / public reference scale. Không cộng token vào utility. Provider có thể gửi A nên price mua cơ hội nhận, không bảo đảm packet hữu ích.

Public scale là median positive relative loss improvement theo absolute round, lấy từ clean monitor của **hai seed còn lại**, pooled modes/actors/roots. Không dùng eval-run logs, market/audit hoặc D counterfactual tại runtime. Đây là cross-fit theo seed/run, **không image-disjoint** giữa các world cùng MNIST train; các folds/calibrations chồng nhau nên không phải independent repetitions của một fixed public scale.

Run gồm binary/magnitude × monitor/K2/K8 trên 18 roots: **108 cây/9,180 state/9,072 transition**, **1,080 fixed-policy rollout20 + 18 local-only controls**. Same service splits, root models, schedules hai actor, utility/cost/attack. Elapsed trong chương trình khoảng 443.5 giây, không phải benchmark overhead.

**Clean all-D final audit Q_net:**

| Mode/K | Monitor (%) | Binary ledger (%) | Magnitude ledger (%) | Magnitude delivered |
| --- | ---: | ---: | ---: | ---: |
| IID/K2 | 89.28 | 89.31 | 89.16 | 203/360 |
| IID/K8 | 89.28 | 89.31 | 88.89 | 75/360 |
| Non-IID/K2 | 87.54 | 87.54 | 87.41 | 339/360 |
| Non-IID/K8 | 87.54 | 87.54 | 87.40 | 328/360 |

Local-only Q_net≈87.52% IID/70.98% non-IID. Magnitude còn lợi ích học chung, nhưng quota cắt nhiều clean IID packets; không được dùng chỉ quality hoặc số A không gửi được để claim incentive success.

**Always-A gain J so all-D cùng condition, peer always-D, gamma=0.95, K8:**

| Mode | Ledger | Actor0 mean gain | Actor0 positive | Actor1 mean gain | Actor1 positive |
| --- | --- | ---: | ---: | ---: | ---: |
| IID | Binary | 0.01823 | 4/9 | 0.08460 | 7/9 |
| IID | Magnitude | 0.05399 | 7/9 | 0.06375 | 8/9 |
| Non-IID | Binary | 0.15386 | 4/9 | 0.29216 | 8/9 |
| Non-IID | Magnitude | 0.23371 | 6/9 | 0.30367 | 8/9 |

Magnitude không dominates binary; particular actor hoặc damage metric có thể cải thiện nhưng strategic gain không giảm đồng đều. Chẳng hạn non-IID/K8, final Q_net dưới node0-always-A tốt hơn ở magnitude, nhưng gain của actor0 vẫn cao hơn. Không suy attack-rate của learner từ các fixed schedules này.

**Quota recovery sau ba A rồi D, IID:**

| K | Binary chưa restore trong window | Magnitude chưa restore trong window |
| --- | ---: | ---: |
| 2 | 0/18 | 10/18 |
| 8 | 0/18 | 9/18 |

Một magnitude case đã restore cần tới 17 D transitions. Các số censored chỉ nói không đạt lại một-slot eligibility, không phải mất mọi probe receipt hoặc blacklist vĩnh viễn. Non-IID magnitude không censored ở các burst cases, có nhiều case còn eligible ngay khi quay lại D; không tính chúng thành restoration delay0.

**Calibration/allocation diagnosis:** ở monitor all-D (chưa gate), median magnitude earnings≈0.247 token/packet IID và≈4.027 non-IID; 335/360 clean IID kiếm dưới giá1, còn 303/360 non-IID kiếm ít nhất1. Shared scale pooled modes tạo allocation rất lệch trong candidate này; không chứng minh mọi magnitude scheme đều thất bại.

**Counterexample diagnostics:** monitor trên model-state cũ phân biệt D≈2.551 tokens với A≈0.167, nhưng enforced magnitude ledger chặn cả packet D/A ở state đó vì balance chiều đối ứng≈0.224<1. Không gọi đó là targeted fix. Case khác, non-IID seed7/snapshot0/K8, prefix DA/DD: D kiếm9.71977 tokens và A1.03693, nhưng cap đưa cả hai về cùng balance4; A vẫn tăng payoff và giảm Q_net so D. Hai model counterfactual được retrain; đây là depth2/H3, không policy20 certificate.

Kiểm chứng:

- 21 regression tests: accounting direction, no same-round funding, quantity discrimination, probe/no-debt/cap conservation, held-out run exclusion và missing calibration fail rõ.
- Recompute độc lập 90 public round medians của ba folds; không dùng evaluation-run seed.
- 62,064 directed accounting/conservation checks, 47,968 relative-gain/credit checks, 1,440 non-probe funding bounds, 21,960 market settlements không có token reward, 432 quota recovery/censoring checks.
- 864 H3 best responses khớp independent enumeration128 policies; 36 monitor trees/analyses và 360 monitor model/payoff rollouts khớp reference; 216 terminal prefix/discounted identities; sáu service manifests không đổi.
- Shared-state hooks giữ 4 historical EMA trees và 8 full rollouts đúng mọi field; replay thêm hai earned-ledger rollouts, hai local-only trajectories độc lập và hai cap counterfactual models. Historical checksums không bị sửa, source hashes run mới khớp code.

**Quyết định:** không promote magnitude ledger này thành final defense; giữ làm negative-control candidate. Dừng chỉnh price/cap/scale hậu nghiệm để ép D. Ưu tiên chốt miền policy và learner episodic dài hạn với local observations, so các mechanism đã đóng băng; nếu quay lại ledger phải nêu calibration context/liquidity/saturation conditions rõ. Official test chưa dùng.



### 3.16. Local episodic learner — protocol đóng băng

[Formulation §18](GAME_FORMULATION.md#18-local-episodic-policy-learner-trên-mechanism-đóng-băng), implementation [mnist_local_policy.py](../experiments/mnist_local_policy.py).

Chốt trước main run: ba world seeds7/19/31 × IID/non-IID, chỉ snapshot5; bốn conditions monitor/EMA-K8/binary-K8/magnitude-K8. Giữ utility/cost/attack/aggregation/data như trước. Hai logistic learners từ p(A)=0.5, train từng seat riêng trước frozen D: immediate-reward semi-gradient và discounted-return score gradient, cùng leave-one-out baseline, 16 updates × 4 episode, LR4; evaluate16 action streams disjoint train.

Observation chỉ gồm own link row/own feedback/receipt/action và public time/probe/horizon. Không có private peer row/quality, action label, D counterfactual hoặc audit. Audit không evaluate khi train. Sau train policy đóng băng, đo unilateral-vs-D và joint deployment của hai policy train riêng; **không phải self-play co-learning, online EGT hoặc dynamic equilibrium**. Train/eval vẫn cùng development world/root và seeded SGD; held-out action draws không phải held-out data.

Main run hoàn tất **24 world/conditions, 96 seat/method training runs, 1,536 updates**, gồm 6,144 train episodes và 3,648 evaluation episodes ×20 settlements. Raw [mnist_local_policy.json](../experiments/results/mnist_local_policy.json); [verification/tổng hợp](../experiments/results/mnist_local_policy_verification.json); [verifier tái lập](../experiments/verify_mnist_local_policy.py). Bốn condition shards chạy đồng thời, manifest giữ hash/config; tổng elapsed các worker không phải parallel wall time.

**Non-IID, hai policy train riêng trước D rồi frozen joint deployment**, mean qua ba world seeds và 16 action draws/world; A-condition được pool bằng raw sent-A/opportunity counts của cả hai node:

| Condition | A/opportunity myopic (%) | A/opportunity return (%) | Q_net initial half (%) | Q_net return (%) |
| --- | ---: | ---: | ---: | ---: |
| Monitor | 91.04 | 85.52 | 83.15 | 78.34 |
| EMA/K8 | 87.66 | 73.14 | 83.42 | 80.33 |
| Binary/K8 | 77.13 | 68.31 | 83.82 | 82.36 |
| Magnitude/K8 | 87.37 | 73.94 | 82.66 | 78.82 |

Return update có ít A hơn myopic ở cả bốn conditions nhưng không đưa về D; mechanism binary có partial suppression so monitor, không phải all-D stability. Joint Q thấp hơn initial-half control ở cả bốn conditions: private improvement trước D không bảo đảm tốt khi hai policy cùng thay đổi. Initial half cũng có endogeneity khác về cơ hội gửi nên không dùng raw A-rate thay conditional A.

**IID, cùng evaluation protocol:**

| Condition | A/opportunity myopic (%) | A/opportunity return (%) | Q_net initial half (%) | Q_net return (%) |
| --- | ---: | ---: | ---: | ---: |
| Monitor | 60.83 | 64.64 | 88.96 | 88.62 |
| EMA/K8 | 59.22 | 53.34 | 88.74 | 88.71 |
| Binary/K8 | 52.92 | 54.07 | 88.71 | 88.58 |
| Magnitude/K8 | 51.85 | 55.53 | 88.60 | 88.49 |

Không có kết luận “long horizon luôn giảm A”: IID chỉ EMA giảm trong mean này. Đây là mô tả pilot, không confidence/significance hoặc optimizer convergence claim.

**Learned return unilateral trước D, non-IID, gain J so D/D trong cùng condition:**

| Condition | Actor0 gain J | Actor1 gain J | World mean gain>0 actor0 | World mean gain>0 actor1 |
| --- | ---: | ---: | ---: | ---: |
| Monitor | 0.42992 | 0.58594 | 3/3 | 3/3 |
| EMA/K8 | 0.06567 | 0.26563 | 1/3 | 3/3 |
| Binary/K8 | 0.12690 | 0.31592 | 1/3 | 3/3 |
| Magnitude/K8 | 0.24609 | 0.28986 | 3/3 | 3/3 |

Mean gain trước D cao hơn initial iid-half unilateral control ở cả hai seat/mọi condition non-IID; không phải một best-response certificate. Ít A có thể còn kiếm lợi hơn: actor1/binary return ít A hơn myopic nhưng gain J cao hơn. Không quy giảm frequency thành poisoning mất incentive.

**Opportunity counterexample:** binary/K8/non-IID, actor0 trước D:

| Policy | A/all rounds (%) | Sent A/all rounds (%) | A/outgoing opportunity (%) | Gain J so D/D |
| --- | ---: | ---: | ---: | ---: |
| Always A | 100 | 45 | 100 | 0.12830 |
| Attack-if-can-send | 45 | 45 | 100 | 0.13473 |

Hai policy có model/link/weights/quality trajectory giống hệt nhau ở từng world. Return tăng đúng attack-cost không phải trả ở blocked sends; attack packets không giảm. Baseline này local, không đọc peer grade/state; không hard-mask actions của learners.

Switching/endgame đã đo, không gán role cố định: binary/non-IID/return joint có mean **7.44 switches/actor** trong 20 actions. Mean preterminal A-rate=66.28%, terminal=67.71%; có 19 cặp action liền kề để đếm switch, không tính root zero-history thành action. Terminal settlement giữ trong mọi J; các tần suất này không chứng minh endgame optimal hoặc một punishment sau horizon.

Kiểm chứng:

- 26 regression tests; exact expected-gradient từ stateful two-step enumeration khớp finite difference, bắt discount/reward-to-go/LOO errors và delayed-incentive reversal.
- Recompute độc lập 1,536 gradient batches, max absolute error≈2.48e−16; 267,840 policy decisions khớp local projection, sigmoid và actor/phase-specific action RNG; 72,960 market settlements giữ utility cũ.
- Sáu split manifests và 90 other-seed calibration medians; 96 historical full-rollout controls, 72 always-A/opportunistic paired model paths khớp mọi field ngoại trừ action/cost đã giải thích.
- Retrain EMA/IID và magnitude/non-IID learned joint trajectories qua legacy rollout, 20 records/case; training path với audit index out-of-bounds vẫn giữ trace và root hash. Hidden-peer-row intervention không đổi current actor0 observation/probability/action.

**Giới hạn/quyết định:** một learning-seed101, warmup5 và finite logistic features; cùng development data/seeded SGD giữa train/eval, không held-out-world/image generalization. LR chung không đảm bảo optimization difficulty giống nhau giữa các state encodings. Giữ mechanisms làm đối chứng, chưa chọn final defense. Bước tiếp là learning-seed/initial-p/horizon/world sensitivity rồi mới joint co-learning và policy-level EGT, không retune penalty để ép D.

### 3.17. Learner robustness và zero-shot world transfer — protocol

[Formulation §19](GAME_FORMULATION.md#19-robustness-của-learner-và-chuyển-policy-sang-world-mới), [runner](../experiments/mnist_policy_sensitivity.py), [verifier](../experiments/verify_mnist_policy_sensitivity.py), [prepared worlds](../experiments/results/mnist_policy_worlds.json).

Giữ cả bốn mechanisms/cost/reward. OFAT: learning-seeds101/211/307 ở p=.5/T20 cho cả hai methods; initial-p=.2/.8 và T10/T25 cho return, so basep=.5/T20. Reuse base training §18, evaluate lại trên common phase2/seed901 cho mọi learning seed/variant; không reuse evaluation cũ hoặc train lại base. Giữ LR4,16×4 episodes/seat.

Định trước source→target7→43,19→59,31→71, cùng mode; giữ nguyên source calibration trên target, không target policy update và không chuyển model/ledger. Đã chuẩn bị sáu new-target roots bằng cùng warmup5, reuse sáu source roots. Target seeds không xuất hiện trong training/calibration; ảnh vẫn có thể overlap vì cùng MNIST train universe, không phải final holdout.

Main run hoàn tất đủ **24 source cases + 24 target-transfer cases**, 24,576 train episodes mới, 6,144 gradient updates và 20,832 evaluation episodes. Reuse 96 baseline training runs, không train lại; các checkpoint đã hoàn tất được giữ nguyên, chỉ bù missing cases sau interruption. [Raw master](../experiments/results/mnist_policy_sensitivity.json), [verification/tổng hợp](../experiments/results/mnist_policy_sensitivity_verification.json). Mọi kết luận dưới đây dùng toàn cohort, không đổi hyperparameters theo partial output.

#### Learning RNG: suppression có lặp lại, không đồng nghĩa đã hội tụ

Non-IID joint frozen deployment, mean [min,max] qua learning seeds101/211/307. Mỗi learning seed pool raw sent-A/opportunity counts của ba source worlds và 16 action draws/world; sau đó mới lấy mean/range qua learning seeds. Range là descriptive estimated outcomes, không population CI:

| Condition | Myopic A/opportunity (%) | Return A/opportunity (%) | Return Q_net (%) |
| --- | ---: | ---: | ---: |
| Monitor | 90.61[90.47,90.73] | 83.78[83.33,84.22] | 79.08[78.63,79.41] |
| EMA/K8 | 86.96[86.60,87.55] | 72.15[70.20,73.43] | 80.81[80.57,81.05] |
| Binary/K8 | 77.42[77.03,77.89] | 66.56[65.04,67.67] | 82.49[82.47,82.51] |
| Magnitude/K8 | 86.44[86.20,86.68] | 72.24[71.35,73.17] | 79.81[79.75,79.87] |

Return ít A hơn myopic ở từng learning seed của cả bốn non-IID conditions. IID không cùng chiều: chỉ EMA giảm ở cả ba learning seeds, ba conditions khác đều tăng. Mean A-rate/Q của một seed không được gọi equilibrium hoặc expected fitness đã biết chính xác.

#### Initial preference: ảnh hưởng lớn hơn range qua learning RNG

Return learner, learning-seed101, T20, cùng số updates/episodes; đây là A/opportunity của joint deployment, không fixed policy probability:

| Mode / condition | A/opportunity p=.2 (%) | p=.5 (%) | p=.8 (%) |
| --- | ---: | ---: | ---: |
| Non-IID / Monitor | 76.93 | 83.80 | 87.19 |
| Non-IID / EMA/K8 | 36.62 | 73.43 | 87.79 |
| Non-IID / Binary/K8 | 37.80 | 67.67 | 81.84 |
| Non-IID / Magnitude/K8 | 52.63 | 73.17 | 80.75 |
| IID / Monitor | 24.58 | 61.20 | 84.84 |
| IID / EMA/K8 | 17.47 | 49.55 | 83.78 |
| IID / Binary/K8 | 20.60 | 51.96 | 83.49 |
| IID / Magnitude/K8 | 20.42 | 53.44 | 85.13 |

Không có một learned A-rate độc lập initializer trong pilot này. Initial-p đổi cả exploration và score-gradient conditioning; finite 16-update result không chứng minh basin attraction hoặc converged policy. Không chọn initial-p thấp để claim defense thành công.

#### Horizon: A-rate không đủ để đánh giá damage hoặc endgame

Non-IID return learner, learning-seed101/p=.5; Q là joint-deployment final quality:

| Condition | A T10 (%) | A T20 (%) | A T25 (%) | Q T10 (%) | Q T20 (%) | Q T25 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Monitor | 88.54 | 83.80 | 82.67 | 76.32 | 79.19 | 81.32 |
| EMA/K8 | 72.88 | 73.43 | 72.13 | 74.66 | 80.57 | 77.78 |
| Binary/K8 | 70.43 | 67.67 | 67.31 | 71.87 | 82.49 | 75.78 |
| Magnitude/K8 | 77.13 | 73.17 | 72.03 | 75.13 | 79.82 | 79.19 |

Không suy T dài hơn luôn bảo vệ quality tốt hơn. Binary có A-rate gần nhau ở T20/T25 nhưng quality khác lớn. Horizon còn đổi số transitions, public progress/final flag và endpoint probe phase; không phải matched-compute hoặc riêng patience intervention.

Gain J trước peer D phải so cùng T; raw gain tăng không đồng nghĩa incentive tăng khi return dài hơn. Ví dụ actor1/binary/non-IID:

| Binary/non-IID horizon | Actor1 gain J trước D | Gain / discount mass |
| --- | ---: | ---: |
| 10 | 0.24884 | 0.03101 |
| 20 | 0.33299 | 0.02595 |
| 25 | 0.36217 | 0.02506 |

Terminal conditional rate cần raw denominator. Mỗi row dưới đây có tối đa 96 terminal directed opportunities; các node/round trong một episode không được coi independent samples:

| Mode / condition | T | Terminal outgoing opportunities | Terminal A/opportunity (%) |
| --- | ---: | ---: | ---: |
| Non-IID / Binary/K8 | 10 | 4 | 75.00 |
| Non-IID / Binary/K8 | 20 | 96 | 69.79 |
| Non-IID / Binary/K8 | 25 | 7 | 100.00 |
| IID / Magnitude/K8 | 10 | 0 | NA |
| IID / Magnitude/K8 | 25 | 0 | NA |

NA là không có outgoing opportunity ở terminal, không phải agent chọn D. Không diễn giải tỷ lệ 100% trên rất ít packets thành all-A optimum hoặc convergence.

Diagnostic từ saved trajectories của **chính T25 binary/non-IID/return joint**, mean qua 48 episodes, không rerun model hoặc kéo dài một T20 policy:

| Action round (T25) | Probe | Q_net (%) | Directed deliveries (%) | Positive neighbor weights (%) |
| --- | --- | ---: | ---: | ---: |
| 23 | no | 73.50 | 1.04 | 0.00 |
| 24 | yes | 82.52 | 100.00 | 100.00 |
| 25 | no | 80.71 | 100.00 | 41.67 |
| 29 | no | 75.78 | 7.29 | 1.04 |

Quality/deliveries biến động mạnh quanh probe; không dùng một endpoint ở probe làm bằng chứng quality ổn định. Đây là observed trajectory, chưa tách riêng causal effect của access, local forgetting, action history và horizon-aware policy.

#### Transfer: so với controls của chính target world

Base return policies train trên source, theta frozen trên target; chỉ action-time own feedback/link state thay đổi, không target policy update. Non-IID mean qua ba source→target pairs, 16 eval draws/pair. A/Q là joint deployment; gains J là evaluate từng learned seat riêng trước D, so target D/D cùng condition:

| Condition | Target A/opportunity (%) | Q_DD (%) | Q_initial (%) | Q_learned (%) | Actor0 gain J vs D | Actor1 gain J vs D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Monitor | 83.91 | 87.62 | 85.60 | 83.91 | 0.42087 | 0.55113 |
| EMA/K8 | 75.89 | 87.62 | 85.31 | 84.54 | 0.09229 | 0.07970 |
| Binary/K8 | 68.35 | 87.62 | 85.38 | 85.10 | 0.10126 | 0.19425 |
| Magnitude/K8 | 74.28 | 87.28 | 85.43 | 84.49 | 0.15109 | 0.38459 |

Binary vẫn có partial suppression so monitor, nhưng không mất profitable learned deviation trong mean; joint quality thấp hơn cả target D/D và matched initial-policy control. Không gọi Q tuyệt đối cao hơn source là generalization improvement: độ khó world khác nhau.

Mean không đại diện mọi target. EMA/binary có positive estimated gain ở hai trong ba target worlds cho từng actor; monitor/magnitude ở cả ba. IID EMA có mean learned-deviation gain âm ở cả hai actor nhưng từng world khác dấu: một transferred policy kém D không chứng minh không còn profitable policy hoặc all-D equilibrium. Target worlds chưa vào policy training/calibration nhưng vẫn share official-train image universe; không phải independent final evaluation.

#### Kiểm chứng và quyết định

- 31 regression tests; targeted uncertainty-summary tests pass sau khi thêm matched quality deltas/raw terminal counts. Consolidator từ chối incomplete/duplicate checkpoints mà không tạo partial master.
- 12 split manifests; 72 historical D/D retained-record prefixes; reuse 96 training runs với immutable baseline/source hashes.
- Recompute 6,144 new gradient batches, max absolute discrepancy≈2.31e−16; 1,286,160 policy decisions khớp local observation/sigmoid/actor-phase RNG; 405,000 settlements giữ utility/cost cũ.
- Model replays: initial-p=.2 binary/non-IID, T25 EMA/IID, zero-shot magnitude/target43; retained model/state/quality records và actor traces khớp. Audit exclusion, target warmup reconstruction và neutral-initializer legacy first batch đều pass.

**Quyết định:** giữ các mechanisms làm đối chứng, chưa promote binary/magnitude thành final defense. RNG stability tại một T không thay được initial-p/horizon/world robustness. Bước tiếp là đóng băng episode-level policy game và fitness từ full J, rồi xét policy-level EGT hoặc joint co-learning tách biệt; không dùng một payoff matrix cho DFL training không reset hoặc gọi PG convergence là ESS/dynamic Nash. Không tăng cost, sửa reward hoặc retune probe để cứu bảng đẹp. Official test tiếp tục chưa dùng.

### 3.18. Frozen episode-policy game — protocol

Bốn bundles **D / attack-if-send / myopic / return**, không phải bốn loại node cố định. Learned bundles lấy seat-specific theta của baseline seed101/p=.5/T20, không train lại. Mỗi mode/condition lấy đều source worlds7/19/31; mỗi encounter reset root5, link state/history và chạy20 settlements với gamma=.95. Không đưa target worlds vào game hoặc mở official test.

Đóng đủ16 ordered profiles/world để không suy đối thủ learned bằng những payoff chỉ đo trước D. Reuse bốn deterministic controls và sáu learned phase2 profiles; sáu cross-policy cells còn thiếu dùng16 action draws, seed901/phase2 như sensitivity. Deterministic controls chỉ lưu một exact episode/profile, không tạo pseudo-replications. World có trọng số bằng nhau bất kể số episode/profile.

Giữ **U0[i,j] / U1[i,j]** riêng cho fixed-seat game; full-trajectory J là payoff. Seat0 deviation thay hàng, seat1 thay cột. Pure Nash/regret chỉ trên finite policy set và empirical means; learner train trước D không tự là best response trước mọi bundle.

Auxiliary random-seat policy-selection game dùng **M[i,j]=(U0[i,j]+U1[j,i])/2**; term thứ hai vẫn là focal i tại seat1. Rare-mutant growth M[i,j]−M[j,j] là boundary diagnostic cho episode-reset bundle population, không chứng minh observed EGT convergence hoặc ESS của fixed-identity DFL. Báo paired action-MC SE và signs từng world, không coi episode count là số independent data worlds.

Implementation: [runner](../experiments/mnist_policy_game.py), [independent verifier](../experiments/verify_mnist_policy_game.py), [payoff-orientation regressions](../experiments/test_mnist_policy_game.py). Mathematical scope ở [formulation §20](GAME_FORMULATION.md#20-episode-reset-policy-game-và-invasion-có-scope). Utility, costs, attack, scalar calibration và probe schedule giữ nguyên; không retune để tạo cooperative equilibrium.

#### Hoàn tất và kiểm chứng

Đủ **24 world/condition cases ×16 ordered profiles =384 cells**. Chạy **2,304 cross-policy episodes mới**, reuse nguyên **2,304 learned phase2 episodes +96 deterministic episodes**; tổng4,704 episodes/94,080 settlements. Không có training mới. [Raw game](../experiments/results/mnist_policy_game.json) và [verification/matrices/deviations](../experiments/results/mnist_policy_game_verification.json) giữ provenance source/input/root, actor traces và Q_net curves.

33 regression tests pass, gồm asymmetric focal-role swap và fixed-seat deviation axes. Verifier kiểm188,160 policy decisions,94,080 utility settlements,384 cells cùng exact reuse equality; hai legacy physical replays binary(myopic,return)/magnitude(attack-if-send,return) non-IID seed7 đều khớp20 bước. Trace vẫn exact khi audit không truy cập được, roots bất biến. Đây không phải re-audit credit conservation từ lean logs.

#### Fixed-seat game: attack-if-send không mất lợi thế

Trong **cả8 mode/condition mean games và từng24 world games**, profile **(attack-if-send, attack-if-send)** là unique pure Nash của plug-in empirical matrices trên bốn bundles; mọi alternative unilateral mean payoff tại profile đó thấp hơn. Đây là restricted empirical pure-Nash result, không phải equilibrium trên toàn bộ local policy space hoặc proof về exact expectation từ16 draws.

Không suy attack-if-send strictly dominates ở mọi opponent/state. Ví dụ non-IID monitor, seat0 trước D, đổi return→attack-if-send **giảm mean J≈.03180**; dấu trước một opponent không quyết định Nash tại opponent khác. Không có global convergence claim hoặc kết luận không tồn tại mixed equilibria.

#### Auxiliary random-seat invasion

Mọi dòng là uniform mean qua ba fixed source worlds. ΔJ là full discounted20-step utility, không final accuracy. ± là **một paired action-MC SE**, không confidence interval/data-world uncertainty; invade D dùng hai deterministic profiles nên action-MC error bằng0, không có nghĩa world/generalization uncertainty bằng0.

| Mode / condition | A-if-send invade D: ΔJ | A-if-send invade return: ΔJ ± MC SE | Return invade A-if-send: ΔJ ± MC SE |
| --- | ---: | ---: | ---: |
| IID / monitor | +0.09471 | +0.06766 ± 0.00274 | -0.07732 ± 0.00258 |
| IID / ema_K8 | +0.03636 | +0.02324 ± 0.00232 | -0.03411 ± 0.00222 |
| IID / binary_K8 | +0.05051 | +0.03569 ± 0.00244 | -0.04002 ± 0.00243 |
| IID / magnitude_K8 | +0.06333 | +0.03568 ± 0.00230 | -0.03658 ± 0.00236 |
| Non-IID / monitor | +0.50962 | +0.13923 ± 0.01753 | -0.14375 ± 0.01363 |
| Non-IID / ema_K8 | +0.20892 | +0.08968 ± 0.00763 | -0.10751 ± 0.01004 |
| Non-IID / binary_K8 | +0.24317 | +0.07988 ± 0.00616 | -0.07957 ± 0.00583 |
| Non-IID / magnitude_K8 | +0.26658 | +0.09879 ± 0.00957 | -0.10699 ± 0.00962 |

Attack-if-send invade D và return có positive empirical growth ở **từng3/3 worlds** của mọi dòng. Tại resident attack-if-send, cả D/myopic/return đều có negative estimated growth ở từng world: restricted boundary diagnostic vẫn thiên về poisoning ở mọi permitted opportunity. Đây không phải quan sát một population thật hội tụ hoặc chứng minh ESS của physical DFL.

Role averaging có thể che seat asymmetry. IID/EMA, đổi return→attack-if-send trước return có seat0 mean gain≈−.00008, seat1≈+.04656; random-seat mean dương không đồng nghĩa mọi node đều có profitable deviation. Non-IID thì cả hai seat có positive mean gain cho deviation này trong từng3/3 worlds ở mọi condition.

Ví dụ **binary/non-IID random-seat M**, raw J, hàng=focal bundle và cột=opponent:

| Focal \ opponent | D | attack-if-send | myopic | return |
| --- | ---: | ---: | ---: | ---: |
| D | 5.48226 | 5.29321 | 5.30090 | 5.28266 |
| attack_if_send | 5.72543 | 5.53558 | 5.57269 | 5.58499 |
| myopic | 5.70243 | 5.48057 | 5.51784 | 5.52925 |
| return | 5.70508 | 5.45601 | 5.49289 | 5.50511 |

Không đối xứng hóa M: focal-role average không làm game thành common-payoff. Raw U0/U1 cho tất cả profiles cùng fixed-seat regret/deviation arrays ở verification artifact; J/Z và per-world invasion gains cũng được giữ.

#### Quality theo cả trajectory: endpoint đẹp không đủ

Mean Q dưới đây là **unweighted time mean của20 settled Q_net**, không fitness hoặc discounted utility. Final là sau action-round24, đúng một probe round. A-if/A-if có A tại100% outgoing opportunities ở mọi condition; all-round A có thể thấp vì không gửi được, không phải tự nguyện hợp tác.

| Non-IID condition | Mean Q qua 20 settlements D/D (%) | Mean Q A-if/A-if (%) | Mean Q return/return (%) | Final Q A-if/A-if (%) | Final Q return/return (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| monitor | 85.96 | 74.68 | 76.77 | 76.50 | 79.19 |
| ema_K8 | 85.96 | 72.91 | 75.22 | 80.42 | 80.57 |
| binary_K8 | 85.96 | 72.90 | 74.72 | 80.90 | 82.49 |
| magnitude_K8 | 84.80 | 73.28 | 76.17 | 75.77 | 79.82 |

EMA/binary cải thiện final Q so monitor ở profiles này, nhưng mean Q qua trajectory **thấp hơn monitor**. Không được dùng probe endpoint để claim uniform damage reduction. Return/return vẫn tốt hơn attack-if/attack-if về time-mean Q trong từng condition, nhưng không đứng vững trước attack-if deviation trong restricted non-IID game.

**Quyết định:** cơ chế hiện tại chưa đạt deterrence ngay cả với một local opportunity-aware baseline. Không promote binary/magnitude, không tăng cost hoặc retune probe để cứu kết quả. Game đã đóng đầy đủ; bước tiếp kiểm cost-gap/attack/tau sensitivities và scoped policy selection/joint adaptation, giữ reset game tách khỏi continuous-state DFL. Official test chưa dùng.

### 3.19. Redesign utility: ký quỹ tuân thủ có audit — proposal, không deployed result

**Đổi ưu tiên:** redesign mechanism và kiểm economic/observability assumptions trước; không chạy thêm EGT/co-learning trên mechanism đã có profitable opportunity-aware deviation chỉ để tìm outcome đẹp. Yêu cầu mới cho phép nhánh reward/payoff khác; baseline market utility/results cũ không sửa ngược.

Chọn candidate **market profit − enforceable collateral forfeiture − fixed audit/funding cost**. Bond không phải reward cho D hoặc tăng physical attack cost. Node giữ quyền sở hữu collateral tới certified fault; refund/unlock principal không phải performance income. Sau first certified fault: seize bond một lần, không nhận thêm outgoing messages của identity đó trong episode, không probe/reset account để né liability.

[Formulation §21](GAME_FORMULATION.md#21-đề-xuất-utility-mới-ký-quỹ-tuân-thủ-có-audit) đặc tả committed data/state/SGD/payload, post-commit unpredictable audit, private-data execution proof, signed evidence/availability, local verification và neutral financial settlement. **Đây là enforcement/trust substrate mới**, không phải heuristic local-validation reputation đã có. Proof/escrow chưa triển khai hoặc benchmark; không dùng hidden simulator action label làm detector.

Conditional theorem: nếu honest canonical execution không bị false seizure, mọi noncanonical send có history-uniform detection floor p_min>0, collateral funded/collectible, base costs/fees không né được khi bị loại, thì `p_min*F >= Z_T+eta`, `Z_T=sum gamma^t`, đảm bảo mọi actual message-deviation policy kém always-compliant sau entry. Proof chặn toàn bộ possible remaining market gain bằng Z_h tại first deviation, không giả định cắt connectivity luôn gây thiệt hại cho attacker. Bao gồm final action; không tự suy model-quality guarantee, ESS, voluntary participation hoặc Byzantine robustness.

Có thêm conservative noisy-audit corollary cho single-bond contract: nếu detection floor là d_min, false-seizure ceiling của honest D mỗi round là f_max, đều uniform conditional trên history, thì sufficient condition là `d_min>f_max*Z_T` và `F >= (Z_T+eta)/(d_min−f_max*Z_T)`. Điều này tính cả future false flags của always-D; không chỉ dùng detector gap d−f của một round. Illustrative d=.25/f=.001 cho F≈54.13963; f=.02 làm certificate này thất bại, không chứng minh deterrence bất khả thi. Rates chưa được đo; participation phải trừ cả honest false-seizure risk.

T20/gamma=.95/eta=.01 cho single-episode bonds12.84028/25.68056/51.36113 tại detection floors1/.5/.25. Đây là conservative illustrative bound, không chosen economic stake; phải kiểm affordable collateral, financing/verification costs và `J_clean−K >= J_outside`. Không giả audit cost=0 hoặc tăng fine tùy ý để có success claim.

**Current economic counterexamples:** full equal revenue pooling vẫn có26/144 positive frozen unilateral gains; clean monitor/non-IID total profit so same-root both-local-only giảm≈.19256/.39107/.02657 ở seeds7/19/31. Với các fixed trajectories/outside reference đó, budget-balanced side payment không đủ làm cả hai prefer clean cooperation, ngay trước audit costs. Không diễn giải thành impossibility của mọi W/market model; cần sửa clean collaboration economics hoặc scope consortium membership đã commit.

[Numerical analysis](../experiments/mnist_incentive_certificate.py), [boundary regressions](../experiments/test_mnist_incentive_certificate.py), [certificate/economic diagnostics](../experiments/results/mnist_incentive_certificate.json). Actual CLI đã chạy;39 regression tests pass. Independent pooling arithmetic khớp144 profiles/26 positive; throwaway exhaustive4-round windfall game kiểm32,767 nontrivial adaptive-before-audit policies đều negative gain tại sufficient bond, largest≈−.01. Noisy single-seizure finite-horizon check giữ negative conservative gain ở H1..20, terminal included. Đây là kiểm theorem/witness, không MNIST rollout với detector thật. Pre-announced audit counterexample vẫn profitable vì chỉ attack lúc không audit. Official test chưa dùng.

**Acceptance gates trước implementation/claim:** có verifiable execution/settlement backend đúng threat model và actual arithmetic; đo completeness/soundness/availability/latency/cost; không phạt honest non-IID vì validation mismatch; kiểm capital/participation feasibility; giữ final economic audit settlement; sau đó mới implement và retrain adaptive policies, kiểm useful sharing/trajectory quality. Nếu chỉ giữ private local-validation signals và không thêm enforcement, phải dùng noisy-monitoring continuation IC có điều kiện, không mượn perfect-audit theorem để quảng cáo bảo đảm.

## 4. Việc tiếp theo — thiết kế defense từ kết quả game

### Mốc 1 — Game hai node có payoff/transition thực

- [x] Tạo state khởi đầu/snapshot MNIST được định danh: 0/5/10 clean-uniform rounds, IID/non-IID và 3 seed.
- [x] Liệt kê đủ bốn action profile từ cùng state và cùng seeded training; chạy uniform, W0 và self-only.
- [x] Ghi model hashes/transition, weights, market quality, audit quality và payoff settlement; tách full-state oracle khỏi observation local.
- [x] Tính stage best responses, pure/mixed Nash và Delta u; kiểm cost-gap=0/tăng, tau cố định và self-only, có xử lý equilibrium continuum.
- [x] Mở cây 1–3 bước; giải exact seeded oracle best response trước bốn policy đối thủ; kiểm chứng độc lập bằng contingent-policy enumeration.
- [x] Công bố scope: chỉ stage Nash và best response nhiều bước; **chưa giải dynamic Nash**, nên chưa chọn equilibrium-selection rule cho game động.

**Đầu ra:** chương trình game nhỏ tái lập, bảng payoff/transition đo từ model thật và kiểm tra phép tính độc lập.

### Mốc 2 — Replicator/invasion và thiết kế W có cơ sở incentive

- [x] Đóng băng stage game/market khi phân tích replicator; không dùng một ma trận cho toàn bộ training.
- [x] Xét điểm biên/nội tại cô lập, A-invasion và Jacobian theo node; giữ giới hạn ở nonhyperbolic/continuum, không tự suy ESS toàn mạng.
- [x] Xác định W0 thường chỉ giảm damage; 3 DD cấu hình gốc đều là self-only. Không tăng cost giả để biến sensitivity thành success claim.
- [x] Thiết kế/chạy W1 local-validation mixture trên cùng snapshot; kiểm Delta u/Delta Q, chia sẻ hợp lệ và quality. Kết quả âm: chưa răn đe hữu ích, clean non-IID giảm so W0.
- [x] Dựng/chạy reputation–conditional-access với history/timing rõ, monitor ablation, 36 cây và 180 rollout; kiểm chứng sanction trễ và honest non-IID exclusion.
- [x] Chạy validation service với cùng budget từ reserved train-validation pool; kiểm split provenance, clean exclusion, quality và paired deviation gains; official test không dùng.
- [x] Align credit/aggregation: chạy realized-gate/mix, kiểm model thực/credit, hai validation assumptions và monitor/access ablation; chưa đạt deterrence.
- [x] Sweep access/probe K1/2/4/8 với candidate cố định, deviations hai actor, recovery/censoring và endgame; có partial deterrence nhưng binary credit/endgame còn lỗ hổng.
- [x] Triển khai/kiểm binary/magnitude earned ledgers với calibration loại eval-run seed, monitor/local controls; accounting đúng nhưng magnitude không tốt hơn đồng đều và quota recovery IID xấu.
- [x] Chạy local episodic myopic/return learners trên mechanisms đóng băng; attack-if-can-send, conditional A-rate, switching/quality/endgame có evidence; không gọi PG là EGT hoặc equilibrium.

**Đầu ra hiện có:** candidate families, local behavior/OFAT/zero-shot evidence và full-J episode-policy game đã kiểm; audit-backed collateral utility là proposal có conditional deterrence proof, chưa operational verification. Ưu tiên observability/enforcement và participation feasibility của redesign, không retune penalty/probe hoặc chạy thêm EGT để cứu mechanism cũ.

### Mốc 3 — Đánh giá cơ chế trên mạng và đóng băng protocol final

- [x] Chạy learning-seed/initial-p/horizon sensitivity và new-world policy transfer trên game hai node, controls cùng utility/cost; evidence ở mục3.17, không coi frozen joint deployment là co-learning.
- [x] Dựng episode-reset policy game từ bốn frozen local bundles/full-trajectory J; đủ ordered matchups, fixed-seat Nash/regret và auxiliary random-seat invasion, giữ riêng scope của continuous DFL.
- [x] Đề xuất audit-backed collateral utility với first-deviation proof, terminal liability, information/enforcement assumptions và capital/participation constraints rõ; numeric certificate không phải deployed result.
- [ ] Kiểm feasibility của verifiable execution/settlement, measured audit/financing overhead và positive private cooperation surplus trước khi implement/promote redesigned mechanism.
- [ ] Xét explicit policy-selection dynamics hoặc joint adaptation với process/update rule công bố riêng; boundary invasion hiện tại không phải observed co-learning/EGT convergence.
- [ ] Mở rộng cost-gap/attack-scale/tau sensitivity và các seed/data/topology regimes; OFAT nhỏ hiện tại chưa là full factorial hoặc broad generalization.
- [ ] Báo Q_net theo thời gian/final, Q_min, từng node/nhóm, clean degradation, outside share, utility, attack rate, switching và compute/communication overhead.
- [ ] Chọn protocol và W bằng development data; tăng seed theo pilot variability và tài nguyên; khóa trước khi đánh giá official test.
- [ ] Sau khi khóa, mới dùng official MNIST test 10,000 ảnh cho final evaluation; không dùng nó tạo market share hoặc chọn tham số.
- [ ] Mở rộng topology, số node, non-IID severity hoặc attack khác từng yếu tố một sau khi claim tối thiểu được kiểm chứng.

**Điều kiện hoàn tất:** chứng minh riêng khả năng bảo vệ model và thay đổi incentive trong miền đã công bố. Không yêu cầu bắt buộc all-D, accuracy tăng trong mọi case hoặc một equilibrium duy nhất.

## 5. Cách chạy lại pilot

Từ root repository, cần Python 3.12 và uv. Môi trường chạy hiện tại thiếu ensurepip nên dùng uv thay cho python -m venv:

```bash
uv venv --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements-pilot.txt
.venv/bin/python experiments/mnist_pilot.py --loss-tolerance 1.0 --output experiments/results/mnist_pilot.json
```

Command này chạy lại 42 run và ghi đè artifact output đã chỉ định. Source/config/data checksum/index split được ghi trong JSON. Cache dữ liệu và virtualenv nằm trong .gitignore; log pilot được giữ làm evidence. Default tolerance của script vẫn là 0.1 để giữ cấu hình reference ban đầu; **phải dùng --loss-tolerance 1.0 cho cấu hình đã chọn**.

Chạy game nhỏ và regression checks từ cùng root:

```bash
.venv/bin/python -m unittest experiments.test_mnist_game
.venv/bin/python -m experiments.mnist_game
.venv/bin/python -m experiments.mnist_game --mechanisms uniform gate self validation_mix --output experiments/results/mnist_w1_game.json
```

Chạy extension reputation và regression checks:

```bash
.venv/bin/python -m unittest experiments.test_mnist_game experiments.test_mnist_reciprocity
.venv/bin/python -m experiments.mnist_reciprocity
.venv/bin/python -m experiments.mnist_reciprocity --validation-modes train_matched service --output experiments/results/mnist_service_validation.json
.venv/bin/python -m experiments.mnist_reciprocity --validation-modes train_matched service --rules realized_gate realized_mix --output experiments/results/mnist_realized_credit.json
```

Chạy access sensitivity với candidate cố định:

```bash
.venv/bin/python -m unittest experiments.test_mnist_game experiments.test_mnist_reciprocity experiments.test_mnist_access_sweep
.venv/bin/python -m experiments.mnist_access_sweep
```

Command đọc reference `mnist_realized_credit.json`, chỉ dùng realized-mix/service và ghi artifact riêng `mnist_access_sweep.json`. `--probe-periods`, `--seeds`, `--modes`, `--snapshot-rounds`, `--horizon`, `--rollout-rounds` chọn subset/sensitivity; default sweep đã được công bố, không phải final-K selection.

Chạy cross-fitted earned-access ledger và controls:

```bash
.venv/bin/python -m unittest experiments.test_mnist_game experiments.test_mnist_reciprocity experiments.test_mnist_access_sweep experiments.test_mnist_earned_access
.venv/bin/python -m experiments.mnist_earned_access
```

Command đọc `mnist_access_sweep.json`, dùng service/realized-mix, calibration theo other-run seeds, binary/magnitude K2/K8 và local-only controls; ghi `mnist_earned_access.json`. Calibration phải phủ mọi absolute round được yêu cầu; không fallback sang evaluation data nếu thiếu.

Chạy local policy study và independent arithmetic + physical model verification:

```bash
.venv/bin/python -m unittest experiments.test_mnist_game experiments.test_mnist_reciprocity experiments.test_mnist_access_sweep experiments.test_mnist_earned_access experiments.test_mnist_local_policy
.venv/bin/python -m experiments.mnist_local_policy
.venv/bin/python -m experiments.verify_mnist_local_policy
```

Runner đọc access-sweep roots/calibration reference; default đủ bốn conditions, hai mode/ba seeds, root5 và predeclared training/eval counts. `--conditions`/`--seeds`/`--modes`/`--output` hỗ trợ shards/subsets; artifact đã công bố là merge strict-equality của bốn condition shards. Verifier đọc raw policy + access/earned references, kiểm số học/controls và retrain hai canonical mode/condition cases nếu có trong subset; kết quả replay ngoài subset được ghi rõ không chạy. Official test không dùng.

Chạy robustness/transfer từ một prepared-worlds manifest cố định:

```bash
# Chỉ chuẩn bị lần đầu nếu chưa có matching worlds artifact:
.venv/bin/python -m experiments.mnist_policy_sensitivity --prepare-worlds
# Có manifest rồi thì reuse, không tạo lại roots hoặc train lại base:
.venv/bin/python -m experiments.mnist_policy_sensitivity
.venv/bin/python -m experiments.verify_mnist_policy_sensitivity
```

Runner giữ nguyên optimizer của baseline để reuse training, hỗ trợ condition/seed/mode subsets. Khi bị ngắt, giữ checkpoint đã hoàn tất và chạy chỉ những source/mode/condition còn thiếu vào output khác. [Consolidator](../experiments/merge_mnist_policy_sensitivity.py) kiểm source/config/world hashes, cấm duplicate source/transfer pairs và chỉ ghi master khi đủ 24 pairs cùng mọi variant:

```bash
python3 -m experiments.merge_mnist_policy_sensitivity --inputs \
  experiments/results/mnist_policy_sensitivity_{monitor,ema_K8,binary_K8,magnitude_K8}.json \
  experiments/results/mnist_policy_sensitivity_resume_{monitor,ema_K8,binary_K8,magnitude_K8}.json \
  experiments/results/mnist_policy_sensitivity_resume_monitor_seed7_non_iid.json \
  experiments/results/mnist_policy_sensitivity_final31_non_iid_{monitor,ema_K8,binary_K8,magnitude_K8}.json
.venv/bin/python -m experiments.verify_mnist_policy_sensitivity
```

Các command có thể ghi đè output được chỉ định; không chạy lại phần compute chỉ để đổi source checksum. Baseline/source snapshot cũ giữ nguyên, verifier §18 dùng archived source đã kiểm digest khi source hiện tại khác; replay không bị thay bằng việc ký lại metadata. Checkpoint elapsed không gồm work bị ngắt trước khi persist và không phải parallel wall time.

Chạy complete episode-reset policy game và independent verification:

```bash
.venv/bin/python -m unittest experiments.test_mnist_policy_game
.venv/bin/python -m experiments.mnist_policy_game
.venv/bin/python -m experiments.verify_mnist_policy_game
```

Runner reuse immutable baseline/sensitivity cells và chỉ evaluate sáu cross-policy cells/world còn thiếu. Checkpoint sau mỗi completed cell; cùng output đã có chỉ resume khi source/input/config khớp, không retrain hay overwrite protocol. Đổi input/source cần output khác. Default game đủ24 cases/384 cells; verifier yêu cầu đủ Cartesian product đã công bố và replay hai canonical non-IID cross-policy episodes. Các matrices là episode-reset empirical game, không payoff matrix cho continuous DFL.

Chạy analytical incentive certificate và economic counterexamples, không train/evaluate neural model mới:

```bash
.venv/bin/python -m unittest experiments.test_mnist_incentive_certificate
.venv/bin/python -m experiments.mnist_incentive_certificate
```

Command chỉ đọc frozen-policy game/baseline và ghi artifact riêng. Full-pool rescoring không retrain policy; bond checks giả định sound execution audit, không biến hidden A labels thành observable detection. `--game`/`--baseline`/`--output` chọn inputs/output; source/input hashes giữ provenance. Không có tiền/ledger/prover backend được triển khai bởi command này.



Command extension đọc root snapshots/config của `mnist_w1_game.json`, ghi artifact output chỉ định. CLI chọn seed/mode/snapshot/horizon/rollout rounds, validation modes, rules và beta/threshold/probe/epsilon. Defaults `train_matched` + `legacy_gate` chạy reference có chủ đích; hai command cuối ghi đối chứng mới riêng. Không có default nào được coi là final-defense selection.

Command game đọc config từ JSON pilot, dùng native split 2 node, lưu snapshots/trees/scenarios trong `experiments/results/`. `--horizon 1..3`, `--snapshot-rounds`, `--seeds`, `--modes`, `--mechanisms` cho phép chạy subset để tái lập. W1 là `validation_mix`, không nằm trong default ba reference mechanisms. Verification artifacts ghi phạm vi kiểm chứng độc lập, không phải chứng nhận defense đã thành công.


## 6. Tiến độ

| Hạng mục | Trạng thái |
| --- | --- |
| Context, utility, timing và information boundary | Đã ghi trong formulation; market/cost còn là giả định cần sensitivity |
| MNIST/split/attack/candidate weight/learner baseline | Đã chọn và chạy pilot |
| Pilot DFL NumPy | 42 development run, có raw logs và split manifests |
| Exact game hai node và best responses dài hạn | 54 cây reference + run mở rộng 72 cây gồm W1, horizon 1–3, kiểm chứng độc lập; không phải deployable local policy |
| Replicator stability/invasion với payoff thực | Frozen stage analysis và full-J episode-reset policy game đã kiểm; attack-if-send vẫn invade D/return trong auxiliary random-seat game. Không có ESS/hội tụ DFL động |
| Defense có đóng góp chiến lược và bằng chứng răn đe | Partial suppression qua learning RNG/new-world transfer, nhưng opportunity-aware poisoning vẫn là restricted empirical equilibrium; endpoint quality không thay trajectory quality. Chưa final defense hoặc co-learning/EGT convergence |
| Final protocol/test evaluation và kết luận | Chưa làm |
