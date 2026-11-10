// Slide deck: tổng hợp thực nghiệm. Build từ repo root:
//   typst compile --root . slides/experiments.typ slides/experiments.pdf
#let accent = rgb("#1f4e79")
#let soft = rgb("#eef3f8")
#let goodc = rgb("#1b7a3a")
#let badc = rgb("#b3261e")
#let res = "../experiments/results/"

#set page(
  width: 33.867cm, height: 19.05cm,
  margin: (x: 1.6cm, top: 1.3cm, bottom: 1.3cm),
  footer: context [
    #set text(size: 11pt, fill: gray)
    DFL poisoning × game theory — tổng hợp thực nghiệm #h(1fr) #counter(page).display() / #counter(page).final().first()
  ],
)
#set text(font: "Noto Sans", size: 17pt, lang: "vi")
#set par(justify: false, leading: 0.6em)
#set list(spacing: 0.55em)
#show table: set text(size: 15pt)
#show raw: set text(size: 0.95em)

#let good(x) = text(fill: goodc, weight: "bold", x)
#let bad(x) = text(fill: badc, weight: "bold", x)
#let note(x) = text(size: 13pt, fill: gray.darken(30%), x)
#let card(body, fill: soft) = block(fill: fill, inset: 10pt, radius: 4pt, width: 100%, body)
#let tbl(..args) = table(stroke: 0.5pt + gray, inset: 5pt, align: center + horizon, ..args)

#let slide(title, body) = {
  pagebreak(weak: true)
  block(width: 100%, below: 0.6em)[
    #text(size: 25pt, weight: "bold", fill: accent)[#title]
    #v(-0.5em)
    #line(length: 100%, stroke: 1.2pt + accent)
  ]
  body
}
#let section(title, sub) = {
  pagebreak(weak: true)
  set page(footer: none)
  align(center + horizon)[
    #text(size: 34pt, weight: "bold", fill: accent)[#title]
    #v(0.4em)
    #text(size: 18pt, fill: gray.darken(20%))[#sub]
  ]
}

// Lattice 4x4 với các vị trí attacker được đánh dấu.
#let lattice(marks) = {
  let cell(i) = {
    let m = marks.at(str(i), default: none)
    box(width: 1.25cm, height: 1.25cm, radius: 50%,
      fill: if m == none { white } else { m },
      stroke: 1pt + accent,
      align(center + horizon, text(size: 12pt, weight: "bold")[#i]))
  }
  grid(columns: 4, column-gutter: 0.55cm, row-gutter: 0.55cm,
    ..range(16).map(cell))
}

// ───────────────────────────── Title
#{
  set page(footer: none)
  align(center + horizon)[
    #text(size: 32pt, weight: "bold", fill: accent)[Chống tấn công đầu độc mô hình trong\ Học liên kết phi tập trung (DFL)\ dựa trên lý thuyết trò chơi]
    #v(0.8em)
    #text(size: 20pt)[Tổng hợp các thí nghiệm đã chạy và kết quả]
    #v(0.4em)
    #text(size: 15pt, fill: gray.darken(20%))[Setup · Utility · Mô hình trò chơi · B1 → B7 · Kiểm tra all-D Nash]
    #v(1.2em)
    #text(size: 14pt, fill: gray)[11/2026 · nguồn: `ROADMAP.md`, `GAME_FORMULATION.md`, `experiments/`, `experiments/results/`]
  ]
}

// ───────────────────────────── Overview
#slide[Câu hỏi nghiên cứu và hai giai đoạn thực nghiệm][
  #card[
    *Câu hỏi:* trong DFL giữa các node cạnh tranh, có thiết kế được *luật nhận/gửi cục bộ W* sao cho đầu độc tham số *mất lợi thế* (không chỉ giảm thiệt hại), và vẫn giữ chất lượng mô hình toàn mạng?
  ]
  #v(0.4em)
  #grid(columns: (1fr, 1fr), column-gutter: 1cm,
    card(fill: rgb("#fdf3e7"))[
      *Giai đoạn 1 — game hai node, utility thị trường* \
      #note[`ROADMAP.md` §3, `GAME_FORMULATION.md`, code ở `../dacn/experiments/`]
      - Ring 4 node / cặp 2 node, hành động rời rạc D/A
      - Payoff = thị phần khách hàng (softmax) − chi phí
      - Stage Nash, replicator, best response 1–3 bước, policy game
      - Thử W0 gate, W1 mixture, reputation, ledger, learner
    ],
    card(fill: rgb("#e9f5ec"))[
      *Giai đoạn 2 — chuỗi B1→B7 + all-D, utility theo λ* \
      #note[`experiments/dfl_lambda.py`, `b4_nash.py`, `alld_check.py`]
      - Lattice 4×4 (16 node), MNIST non-IID
      - Cường độ tấn công liên tục λ, utility tuyến tính trong q
      - Best response 1 attacker → Nash N người → all-D Nash
      - So sánh 7 defense, thiết kế W2 (gate + trừng phạt có ân xá)
    ],
  )
  #v(0.3em)
  Kết luận ngắn: giai đoạn 1 cho thấy *giảm thiệt hại ≠ răn đe*; giai đoạn 2 đo răn đe trực tiếp bằng *chi phí tấn công tới hạn* $c^*$ và tìm được W2 làm $c_A^*$ giảm ≈ 9–16× so với uniform.
]

// ───────────────────────────── Phase 1
#section[Phần I — Game hai node, utility thị trường][Giai đoạn 1 (ROADMAP §2–§3): bằng chứng âm dẫn tới thiết kế lại]

#slide[Giai đoạn 1: setup và mô hình trò chơi][
  #grid(columns: (1.1fr, 1fr), column-gutter: 0.8cm,
    [
      *Setup*
      - MNIST, MLP 784→64→10, SGD lr 0.1, batch 64, 1 epoch/vòng
      - 1024 train + 200 val/node; non-IID Dirichlet $alpha = 1$
      - Tấn công sign-flip khuếch đại (scale 5):\ $m_i^t = w_i^t - 5(ell_i^t - w_i^t)$, model cục bộ vẫn sạch
      - Mỗi vòng mỗi node chọn *D* (gửi sạch) hoặc *A* (đầu độc)
      *Utility (thị trường)*
      $ s_(i g)^t = (exp[kappa(q_(i g)^t - tau_t)]) / (1 + sum_j exp[kappa(q_(j g)^t - tau_t)]) $
      $ macron(u)_i^(t+1) = sigma_i^(t+1) - macron(c)_i^t, quad sigma_i = sum_g mu_g s_(i g) $
      #note[$kappa = 10$, $tau_t = min(0.8, 0.1 + 0.05 t)$, $c_D = 0.005$, $c_A = 0.006$, $gamma = 0.95$, $T = 20$]
    ],
    [
      *Mô hình trò chơi*
      - Stage game 2 người (DD, AD, DA, AA), Nash thuần/hỗn hợp
      - Replicator: $dot(x) = x(1-x)[f_A (x; W) - f_D (x; W)]$; A xâm nhập nếu $f_A (0^+) > f_D (0^+)$
      - Best response chính xác: cây trạng thái, 128 policy tất định, chân trời 1–3 bước
      - Policy game: 4 bundle đóng băng × 16 profile
      *Luật trọng số thử nghiệm*
      - W0 gate: nhận $m_j$ nếu $L_i (m_j) <= L_i (ell_i) + 1.0$
      - W1: trộn $lambda in {0, .125, .25, .5}$ theo val-loss
      - Reputation + access, earned-access ledger, local learner
    ],
  )
]

#slide[Giai đoạn 1: kết quả chính (đều âm về răn đe)][
  #tbl(columns: (auto, 1.6fr, 2.2fr, auto), align: (center, left, left, center),
    [*§*], [*Thí nghiệm*], [*Số liệu*], [*Kết luận*],
    [3.2], [Pilot 42 run, ring 4 node], [non-IID: sạch 87.95%, A thích ứng + W0 80.43%; A-rate ≈ 0.50], [giảm thiệt hại,\ không răn đe],
    [3.6], [Stage Nash, 18 snapshot × 3 cơ chế], [uniform: 18/18 AA; W0: 12 AA, 3 AD/DA, 3 DD (DD = loại hết hàng xóm)], bad[âm],
    [3.8], [W1 mixture, 72 cây H1–3], [A vẫn có trọng số dương 156/1512; xâm nhập đơn phương có lợi 16/36], bad[âm],
    [3.9], [Chất lượng sạch W1], [non-IID mất 4.22 pp so với W0, chỉ hơn local-only 0.82 pp], bad[âm],
    [3.10], [Reputation–reciprocity], [always-A vẫn lãi +0.333; loại nhầm node trung thực], bad[âm],
    [3.14], [Probe K = 1/2/4/8], [always-A vẫn dương; endgame: A ở vòng cuối có lợi 9/9 root non-IID], [chưa kết luận],
    [3.18], [Policy game 4 bundle, 4704 episode], [(attack-if-send)² là Nash thuần duy nhất; chất lượng 74.68% vs D/D 85.96%], bad[âm],
    [3.19], [Utility ký quỹ có audit], [định lý điều kiện; chưa triển khai], [đề xuất],
  )
  #v(0.3em)
  #card[*Bài học:* (1) lọc tốt không làm tấn công bất lợi; (2) utility thị trường + hành động nhị phân khó tách tác động; (3) cần đo *incentive* trực tiếp, trên mạng nhiều node, với cường độ tấn công liên tục → Phần II.]
]

// ───────────────────────────── Phase 2
#section[Phần II — Chuỗi thí nghiệm B1 → B7 → all-D][Utility tuyến tính theo chất lượng, cường độ tấn công λ, mạng 16 node]

#slide[Setup chung (mọi thí nghiệm B1–B7, all-D)][
  #grid(columns: (1.5fr, 1fr), column-gutter: 0.8cm,
    tbl(columns: (auto, 1fr), align: (left, left),
      [Dữ liệu], [MNIST 60k official-train; *official test không bao giờ được nạp*],
      [Chia tách], [80% train pool / 100 ảnh/lớp audit cân bằng (1000) / phần còn lại làm val; kiểm tra không rò rỉ],
      [Non-IID], [Dirichlet $alpha = 1.0$ trên nhãn; 1024 train + 200 val mỗi node],
      [Val], [`local` (theo phân phối node) hoặc `balanced` (20 ảnh/lớp, B7)],
      [Mô hình], [MLP 784 → 64 ReLU → 10, SGD lr 0.1, batch 64, 1 epoch/vòng],
      [Topology], [Lattice 4×4, láng giềng von Neumann (bậc 2/3/4)],
      [Thời gian], [$T = 40$ vòng đồng bộ, tấn công từ vòng $s = 10$],
      [Seeds], [search {7,19,31,43,59}; holdout {61,67,71,73,79}],
      [Đo chất lượng], [$q_i$: acc trên val của node; *audit*: acc trên tập cân bằng],
    ),
    [
      #align(center, lattice((
        "0": rgb("#f6c9c5"), "1": rgb("#fbe3b5"), "5": rgb("#c9e4cf"))))
      #v(0.3em)
      #note[Vị trí attacker trong B2/B3/B5–B7: \ *0* góc (bậc 2) · *1* cạnh (bậc 3) · *5* trong (bậc 4)]
      #v(0.3em)
      #card[RNG huấn luyện khóa theo (seed, node, vòng) → các run chỉ khác λ là *cặp* (paired): mọi thiệt hại = clean − attacked cùng seed.]
    ],
  )
]

#slide[Mô hình tấn công: cường độ liên tục λ][
  Mỗi vòng $t$, node $i$ (đang phục vụ model $w_i^t$):
  $ ell_i = "SGD"(w_i^t) quad quad m_i = ell_i - lambda_i^t (ell_i - w_i^t) quad quad w_i^(t+1) = "AGG"_i (ell_i, {m_j : j in cal(N)_i}) $
  #grid(columns: (1fr, 1fr), column-gutter: 0.8cm,
    card[
      - $lambda = 0$: trung thực (gửi $ell_i$)
      - $lambda = 1$: gửi lại $w_i^t$ (free-ride, không đóng góp)
      - $lambda > 1$: đảo chiều + khuếch đại cập nhật (đầu độc)
      - Attacker *giữ model cục bộ sạch* $ell_i$; chỉ message bị thao túng
    ],
    card[
      - Lưới λ: {0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16} (B1–B6)\ {2, 3, 4, 5, 6, 8, 12(, 16)} (B7, all-D)
      - Tấn công tĩnh: λ cố định từ vòng 10 đến 40\ (lớp chính sách giới hạn)
      - Attacker vẫn chạy cùng luật nhận W như mọi node
    ],
  )
  #v(0.3em)
  Hệ quả quan trọng: tấn công làm hại láng giềng *và quay lại hại chính attacker* qua bước tổng hợp vòng sau ("own drop") — đây là kênh răn đe tự nhiên.
]

#slide[Utility và chuẩn hóa][
  $ u_i = alpha_1 q_i + alpha_2 macron(q)_i + alpha_3 (q_i - macron(q)_i) - gamma kappa lambda_i^2 $
  #note[$q_i$: chất lượng của $i$; $macron(q)_i$: trung bình láng giềng; $alpha_3$: lợi ích *vị thế tương đối* (cạnh tranh); $gamma kappa lambda^2$: chi phí tấn công.]
  #v(0.2em)
  Viết lại $u_i = (alpha_1 + alpha_3) q_i - (alpha_3 - alpha_2) macron(q)_i - gamma kappa lambda_i^2$, chia cho $alpha_3 - alpha_2 > 0$:
  #card[
    $ U_i = sum_t w_t [rho thin q_i^t - macron(q)_i^t] - c thin lambda_i^2, quad rho = (alpha_1 + alpha_3)/(alpha_3 - alpha_2) > 1, quad c = (gamma kappa)/(alpha_3 - alpha_2), quad w_t prop 0.95^t $
  ]
  #grid(columns: (1fr, 1fr), column-gutter: 0.8cm,
    [
      *Lợi ích của tấn công* so với trung thực (paired):
      - một lần: $G_1(lambda) = D_1(lambda) - c lambda^2$
      - lặp: $G_R (lambda; rho) = sum_k w_k [D_k - rho "Own"_k] - c lambda^2$
      $D_k$ = sụt $macron(q)$ láng giềng, $"Own"_k$ = sụt $q$ của attacker sau $k$ vòng.
    ],
    [
      *Chi phí tới hạn*
      $ c^* = max_lambda V(lambda) / lambda^2 $
      tấn công có lợi $<=> c < c^*$. Biến thể *chi phí hằng* (B7, all-D):\ $U_i = dots - c_A bb(1)[s_i > 0]$, all-D là Nash $<=> c_A >= c_A^* = max_i max_s G_i (s)$.
      #note[Nếu $alpha_3 <= alpha_2$ thì không λ \> 0 nào có lợi.]
    ],
  )
]

#slide[Mô hình trò chơi và giao thức thống kê][
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.5cm,
    card[
      *B3 — Best response 1 attacker* \
      Mọi node khác trung thực. Tính $V(lambda)$ từ các run cặp B2 (không train thêm), suy ra $lambda^*(c, rho)$ và $c^*$ cho game một lần và lặp.
    ],
    card[
      *B4 — Nash N người* \
      Cả 16 node chiến lược, mỗi node chọn $lambda_i in {0,4,5,6,8,10,12}$. Iterated best response (Gauss–Seidel, node 0→15); hội tụ khi một sweep không có node đổi.
    ],
    card[
      *all-D check* \
      Mỗi node lệch đơn phương $s in {2,…,16}$, 16 × 8 profile. Chọn $s_i$ trên search seeds, ước lượng $G_i (s_i)$ trên *holdout* → không bias chọn lựa.
    ],
  )
  #v(0.4em)
  *Quy tắc ý nghĩa thống kê* (5 seed cặp):
  - Một λ chỉ được tính là có lợi nếu $overline(G) > t_(0.975, 4) dot "SE" = 2.776 dot "SE"$; ngược lại bị loại → nhiễu seed không tạo ra "tấn công có lợi" giả.
  - B4: node chỉ đổi chiến lược nếu cải thiện có ý nghĩa; Nash = *Nash thống kê trên lưới λ*.
  - `verify_nash.py`: kiểm tra lại từng attacker của profile Nash trên 5 holdout seed.
  - Mọi kết quả báo cáo cho cả hai thước đo chất lượng: $q$ (val cục bộ, chính) và *audit* cân bằng (độ nhạy).
]

#slide[Thang thí nghiệm][
  #let step(c, t, b) = box(width: 100%, fill: c, inset: 8pt, radius: 4pt)[*#t* \ #text(size: 13pt)[#b]]
  #let left = align(center + horizon, text(size: 22pt, fill: accent)[←])
  #let arrow = align(center + horizon, text(size: 22pt, fill: accent)[→])
  #grid(columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr), column-gutter: 0.2cm, row-gutter: 0.5cm,
    step(soft)[B1 baseline][DFL sạch vs local-only: hợp tác có lợi không?], arrow,
    step(soft)[B2 + attack][1 attacker, λ ∈ [0.5, 16]: đường thiệt hại D(λ), lan truyền theo hop], arrow,
    step(soft)[B3 incentive][Best response λ\*(c, ρ), chi phí tới hạn c\*], arrow,
    step(soft)[B4 Nash N người][Iterated BR trên 16 node, kiểm tra holdout],
    [], [], [], [], [], [], align(center, text(size: 22pt, fill: accent)[↓]),
    step(rgb("#e9f5ec"))[all-D check][Lệch đơn phương, search/holdout, c#sub[A]\* theo W], left,
    step(rgb("#e9f5ec"))[B7 W2][Val cân bằng, tol hiệu chuẩn 0.3, trừng phạt L vòng + ân xá G], left,
    step(rgb("#fdf3e7"))[B6 + recip][Gate + cắt gửi theo trust; Nash theo metric audit], left,
    step(rgb("#fdf3e7"))[B5 + defense][gate, median, trimmed, clip, BALANCE],
  )
  #v(0.4em)
  #note[Script: `dfl_lambda.py` (B1, B2, B5, B6, B7) · `analyze_b1_b2.py` · `analyze_b3.py` · `b4_nash.py` (B4, B5/B6 Nash) · `verify_nash.py` · `analyze_b5.py` · `analyze_b6.py`, `calib_b6.py` · `analyze_b7.py` · `alld_check.py`]
]

// ───────────────────────────── B1
#slide[B1 — Baseline: DFL sạch so với local-only][
  #tbl(columns: (auto, 1fr, 1fr, 1fr, 1fr),
    [], [*q (val cục bộ)*], [*q min-node*], [*audit cân bằng*], [*audit min-node*],
    [local-only], [90.24 ± 0.66], [85.38 ± 1.51], [77.05 ± 2.06], [65.69 ± 1.42],
    [DFL uniform], [91.88 ± 0.38], [87.99 ± 1.27], good[89.87 ± 0.66], good[88.70 ± 0.71],
  )
  #note[% , trung bình 10 vòng cuối, mean ± sd trên 5 seed]
  #v(0.4em)
  - DFL tăng audit ở *80/80* node-seed (trung bình *+12.82 pp*), $q$ tăng ở 66/80 (+1.64 pp).
  - Node yếu nhất được lợi nhiều nhất: audit min-node 65.7 → 88.7.
  - ⇒ Có *động cơ tham gia* hợp tác; tấn công có giá trị để phá hoại → đáng mô hình hóa.
  - $q$ cục bộ "dễ" hơn (đo trên phân phối lệch của node) nên khoảng cách nhỏ; audit phản ánh chất lượng toàn cục.
]

// ───────────────────────────── B2
#slide[B2 — Thêm attack: đường thiệt hại D(λ) (aggregation uniform)][
  #image(res + "b1_b2_lattice_noniid_b2.png", width: 100%)
  #grid(columns: (1.3fr, 1fr), column-gutter: 0.6cm,
    tbl(columns: 6,
      [*λ* (att. 5)], [*D1*], [*Dss*], [*own*], [*net audit*], [*hop1/2/3+*],
      [1], [−0.45], [−0.18], [0.11], [0.08], [−0.2/0.1/0.1],
      [4], [3.60], [3.40], [1.09], [1.42], [3.4/0.6/0.4],
      [8], [15.05], [20.64], [12.44], [12.97], [20.6/5.1/1.7],
      [16], [36.08], [55.92], [46.16], [45.48], [55.9/26.7/13.0],
    ),
    [
      - λ ≤ 2: không gây hại (nhiễu ±1 pp); ngưỡng ≈ λ 3–4, sau đó *lồi*.
      - Thiệt hại lan theo hop, giảm ≈ ½ mỗi hop.
      - "own" lớn: attacker tự chịu ≈ 60–85% thiệt hại của láng giềng (λ ≥ 8) ⇒ ρ có vai trò răn đe.
      - Node cạnh (bậc 3) gây hại nhiều nhất.
    ],
  )
]

// ───────────────────────────── B3
#slide[B3 — Best response của một attacker (uniform)][
  #image(res + "b3_lattice_noniid_q.png", width: 92%)
  #grid(columns: (1.25fr, 1fr), column-gutter: 0.6cm,
    tbl(columns: 6,
      [*c\* (metric q)*], [*một lần*], [*ρ=1*], [*ρ=1.25*], [*ρ=2*], [*ρ=3*],
      [att 0 (bậc 2)], [1.64e-3], [8.3e-4], [–], [–], [–],
      [att 1 (bậc 3)], [2.79e-3], [1.66e-3], [1.35e-3], [–], [–],
      [att 5 (bậc 4)], [2.38e-3], [1.80e-3], [1.63e-3], [1.18e-3], [–],
      [att 5, audit], [2.23e-3], [2.25e-3], [2.03e-3], [1.51e-3], [–],
    ),
    [
      - *Nhảy bậc*: khi $c$ vượt xuống dưới $c^*$, $lambda^*$ nhảy 0 → 6 (att 5) — không có vùng "tấn công nhẹ".
      - Game lặp + ρ lớn ⇒ tự hại răn đe: ρ = 3 không còn λ có lợi.
      - Theo audit, tấn công có lợi ở ρ cao hơn (own drop đo trên audit nhỏ hơn).
    ],
  )
]

// ───────────────────────────── B4
#slide[B4 — Nash N người (16 node chiến lược, uniform, metric q)][
  #tbl(columns: (auto, auto, 1.6fr, auto, auto, auto),
    [*ρ*], [*c*], [*Profile Nash (λ ≠ 0)*], [*q: all-D → Nash*], [*audit: all-D → Nash*], [*holdout*],
    [1.25], [1e-4], [node 1, 11: λ = 6], [91.88 → 83.30], [89.87 → 79.67], [0/2 còn ý nghĩa],
    [1.25], [1e-3], [all-D], [91.88], [89.87], [—],
    [1.5], [1e-4], [node 5: λ = 6], [91.88 → 87.03], [89.87 → 84.79], [0/1],
    [1.5], [1e-3], [all-D], [91.88], [89.87], [—],
    [2.0], [1e-4, 1e-3], [all-D], [91.88], [89.87], [—],
  )
  #v(0.3em)
  - BR hội tụ trong ≤ 2 sweep; 815–1675 mô phỏng mỗi cấu hình.
  - Chỉ 1–2 attacker xuất hiện, nhưng đã làm $q$ trung bình *cả mạng* mất 4.9–8.6 pp.
  - Kiểm tra trên holdout: lợi ích của các attacker *không còn ý nghĩa* (vd. node 1: −0.047, SE 0.012) ⇒ đây là cân bằng biên/nhiễu, nhưng cho thấy uniform *không chặn* được tấn công khi $c$ nhỏ.
  - Với $c >= 10^(-3)$ hoặc $rho >= 2$: all-D là Nash thống kê → khớp với $c^*$ ở B3.
]

// ───────────────────────────── B5
#slide[B5 — Thêm defense: chi phí khi sạch và thiệt hại][
  #grid(columns: (1fr, 1.25fr), column-gutter: 0.6cm,
    [
      #tbl(columns: 4,
        [*Defense*], [*q*], [*audit*], [*audit min*],
        [uniform], [91.88], [89.87], [88.70],
        [gate (tol 1.0)], [91.90], [89.88], [88.72],
        [median], [91.91], bad[88.49], bad[86.38],
        [trimmed], [91.94], [89.05], [87.30],
        [clip], [91.95], [89.83], [88.63],
        [BALANCE], [92.35], [89.31], [87.93],
        [recip (B6)], [91.85], [89.80], [88.02],
      )
      #note[Clean, 10 vòng cuối. Median/trimmed mất 0.8–1.4 pp audit do non-IID.]
    ],
    [
      #tbl(columns: 4,
        [*Sụt $macron(q)$ láng giềng*], [*λ=4*], [*λ=8*], [*λ=16*],
        [uniform], [3.4/3.2/1.2], bad[20.6/26.2/15.5], bad[55.9/61.4/49.4],
        [gate], [0.8/1.5/0.5], [0.0/0.5/−0.4], [0.0/0.4/−0.4],
        [median], [0.9/1.1/−0.1], [1.4/1.4/0.1], [1.6/1.6/0.2],
        [trimmed], [1.3/1.1/−0.2], [1.9/1.4/0.0], [2.0/1.6/0.0],
        [clip], [0.8/1.3/−0.1], [0.9/3.4/−0.1], [1.0/9.9/−0.2],
        [BALANCE], [0.6/1.1/0.3], [2.2/3.3/1.3], [0.0/0.7/−0.2],
        [recip], [0.7/1.1/0.7], [0.5/0.9/0.3], [0.2/0.6/−0.2],
      )
      #note[pp, attacker 5 / 1 / 0, trung bình các vòng bị tấn công]
    ],
  )
  #v(0.2em)
  Mọi defense đều *chặn được tấn công mạnh* (λ ≥ 8). Vùng còn lại là λ ≈ 3–5: dưới ngưỡng lọc, vẫn gây 0.5–1.5 pp.
]

#slide[B5 — Defense có làm tấn công *bất lợi* không?][
  #grid(columns: (1fr, 1fr), column-gutter: 0.6cm,
    [
      *Chi phí tới hạn c\* (metric q, game lặp ρ = 1.25)*
      #tbl(columns: 4,
        [*Defense*], [*att 5*], [*att 1*], [*att 0*],
        [uniform], [1.6e-3], [1.3e-3], [–],
        [gate], good[–], good[–], good[–],
        [median], good[–], good[–], good[–],
        [trimmed], good[–], good[–], good[–],
        [clip], [–], [4.7e-4], [–],
        [BALANCE], [3.5e-4], [4.7e-4], [–],
        [recip], good[–], [8.4e-4], [–],
      )
      #note["–": không λ nào có lợi có ý nghĩa thống kê.]
    ],
    [
      *Nash N người (ρ = 1.25, c = 1e-4, metric q)*
      #tbl(columns: 4,
        [*Defense*], [*\# attacker*], [*q all-D → Nash*], [*holdout*],
        [gate], good[0], [91.90], [—],
        [median], good[0], [91.91], [—],
        [trimmed], good[0], [91.94], [—],
        [clip], bad[2], [91.95 → 79.96], [0/2],
        [BALANCE], bad[10], [92.35 → 85.91], [0/10],
      )
      #v(0.3em)
      - Clip/BALANCE: lọc theo *khoảng cách* để lọt tấn công trung bình → có profile nhiều attacker.
      - Gate/median/trimmed: all-D là Nash theo $q$.
    ],
  )
  #card[Nhưng metric $q$ (val cục bộ non-IID) chưa phải toàn bộ câu chuyện → B6 kiểm tra lại theo audit.]
]

// ───────────────────────────── B6
#slide[B6 — Defense đề xuất "recip": gate + có đi có lại][
  #grid(columns: (1.1fr, 1fr), column-gutter: 0.6cm,
    [
      *Luật (mỗi node, chỉ dùng thông tin cục bộ)*
      - Nhận: $"excess"_(i j) = L_i (m_j) - L_i (ell_i)$; giữ $m_j$ nếu excess ≤ `gate_tol` (1.0)
      - Trust EMA: $T_(i j) <- beta T_(i j) + (1 - beta) bb(1)["excess" <= "flag_tol"]$ (β = 0.5, flag 0.5 ≈ p99 excess trung thực)
      - Gửi: $j -> i$ chỉ khi $T_(j i) >= 0.75$; không nhận được → trust trôi về 1
      - Cắt một chiều: attacker vẫn gửi ⇒ vẫn tạo bằng chứng ⇒ bị cắt liên tục; hai node trung thực cắt nhầm sẽ hồi phục
    ],
    [
      *Chẩn đoán liên kết* (`analyze_b6.py`)
      #tbl(columns: 4,
        [*λ* (att 1)], [*Gửi tới attacker*], [*Msg attacker qua gate*], [],
        [≤ 2], [100%], [100%], [],
        [3], [52.0%], [92.0%], [],
        [4], [33.1%], [72.9%], [],
        [8], [9.6%], [49.6%], [],
        [16], [3.6%], [12.9%], [],
      )
      #note[Clean: 97.57% liên kết trung thực được giữ (175/7200 bị cắt nhầm).]
    ],
  )
  #card[Cơ chế răn đe: láng giềng *ngừng gửi model cho attacker* ⇒ attacker mất lợi ích hợp tác (B1: +12.8 pp audit) ⇒ "own drop" tăng.]
]

#slide[B6 — Nash theo metric *audit*: gate và recip thất bại][
  #tbl(columns: (auto, auto, 1.4fr, auto, auto),
    [*Defense*], [*Metric*], [*Profile Nash (ρ = 1.25, c = 1e-4)*], [*audit all-D → Nash*], [*holdout còn ý nghĩa*],
    [recip], [q], [node 1: λ = 3], [89.80 → 89.19], good[0/1],
    [recip], [audit], [16/16 tấn công, λ ∈ {2,3,4,8}], bad[89.80 → 81.12], bad[11/16],
    [gate], [audit], [16/16 tấn công, λ ∈ {5,6,8,12}], bad[89.88 → 79.24], bad[14/16],
    [uniform], [audit], [13/16 tấn công, λ ∈ {3,…,12}], bad[89.87 → 10.01], [2/13],
  )
  #v(0.3em)
  - Theo audit, *mọi node đều tấn công* và lợi ích được xác nhận trên holdout (gate: +0.05 … +0.14 mỗi node).
  - Nguyên nhân: gate dùng *val cục bộ non-IID* với tol 1.0 nats → message λ 3–8 vẫn qua gate (49–92%), gây hại theo audit cân bằng nhưng gần như vô hình theo $q$ cục bộ.
  - Uniform/audit: mạng sụp về 10% — trong trạng thái sụp, từng lệch đơn phương chỉ còn là nhiễu (2/13).
  #card[⇒ *Lệch thước đo*: bộ lọc nhìn phân phối khác với thứ utility đánh giá. Sửa ở B7: val *cân bằng* + ngưỡng hiệu chuẩn + trừng phạt tường minh.]
]

// ───────────────────────────── B7
#slide[B7 — Hiệu chuẩn ngưỡng với val cân bằng][
  #grid(columns: (1fr, 1.5fr), column-gutter: 0.6cm,
    [
      *Excess của message trung thực* (n = 7200, t ≥ 10)
      #tbl(columns: 2,
        [*Phân vị*], [*excess (nats)*],
        [p50], [0.0023], [p90], [0.1029], [p99], [0.2185],
        [p99.9], [0.3073], [max], [0.3533],
      )
    ],
    [
      *Tỷ lệ bị gắn cờ (excess \> tol)*
      #tbl(columns: 7,
        [*tol*], [*trung thực*], [*λ=2*], [*λ=3*], [*λ=4*], [*λ=5*], [*λ=8*],
        [0.10], [10.53%], [8.7%], [90.9%], [100%], [100%], [100%],
        [0.20], [1.51%], [0.2%], [75.8%], [98.9%], [99.9%], [100%],
        [*0.30*], good[0.12%], [0.0%], [56.5%], [96.8%], [99.9%], [100%],
        [0.50], [0.00%], [0.0%], [18.0%], [88.0%], [97.7%], [100%],
        [1.00], [0.00%], [0.0%], [0.1%], [37.3%], [82.7%], [99.4%],
      )
    ],
  )
  #v(0.3em)
  - Chọn *tol = 0.3* (≈ p99.9 trung thực): báo động nhầm 0.12%, bắt 97% message λ = 4.
  - λ = 2 *không phát hiện được* ở mọi ngưỡng hợp lý → vùng tấn công tồn dư; may mắn là nó gần như vô hại (B2: ≤ 0.5 pp).
  - Biến thể W: `gate03` (gate tol 0.3), `recipEMA`, `recipL3` / `recipL10` (cắt L vòng sau mỗi cờ), *W2 = `recipL3G2`* (cắt 3 vòng + bắt buộc gửi 2 vòng ân xá → tránh vòng xoáy cắt nhầm).
]

#slide[B7 — Sàng lọc các biến thể W (chi phí tấn công hằng $c_A$)][
  #grid(columns: (1fr, 1.15fr), column-gutter: 0.5cm,
    [
      #tbl(columns: 4,
        [*Biến thể*], [*audit*], [*audit min*], [*own drop s=4*],
        [uniform], [89.87], [88.70], [0.7/0.9/1.0],
        [gate10], [89.87], [88.70], [0.2/0.6/0.6],
        [gate03], [89.88], [88.68], [−0.1/0.1/0.1],
        [recipEMA], [89.59], bad[85.22], good[5.8/3.6/4.9],
        [recipL3], [89.58], bad[85.17], good[7.1/4.8/5.1],
        [recipL10], [89.55], bad[85.16], good[8.8/5.9/6.6],
        [*W2 L3G2*], [89.87], good[88.68], good[3.1/2.0/2.5],
      )
      #note[Clean all-D, val cân bằng; own drop: pp, metric q, attacker 5/1/0]
    ],
    [
      #tbl(columns: 3,
        [*$c_A^* times 100$ (audit)*], [*ρ = 1.25*], [*ρ = 2*],
        [uniform], bad[21.6 / 23.5 / 22.1], bad[9.2 / 8.9 / 11.0],
        [gate10], [0.8 / 1.7 / 1.4], [– / 1.4 / 0.8],
        [gate03], [0.3 / 0.7 / 0.7], [– / 0.6 / 0.7],
        [recipEMA/L3/L10], good[– / – / –], good[– / – / –],
        [*W2 L3G2*], good[– / – / –], good[– / – / –],
      )
      #note[attacker 5 / 1 / 0, search seeds; "–" = không lợi ích có ý nghĩa]
    ],
  )
  #v(0.3em)
  - Trừng phạt (recip) làm attacker *tự mất 2–9 pp* — kênh răn đe mà gate thuần không có.
  - recip không ân xá: cắt nhầm lan rộng → node yếu nhất mất ≈ 3.5 pp. *Ân xá G = 2* giữ nguyên chất lượng node yếu (88.68) mà vẫn răn đe.
]

// ───────────────────────────── all-D
#slide[Kiểm tra cuối: all-D có phải Nash? (search → holdout, 16 node × 8 λ)][
  #tbl(columns: (1.3fr, auto, auto, auto, auto, auto),
    [*W (val)*], [*clean q / audit*], [*$c_A^* times 100$, q*], [*node có lợi (q)*], [*$c_A^* times 100$, audit*], [*node có lợi (audit)*],
    [uniform (local)], [91.20 / 89.73], bad[22.89], [0/16], bad[37.63], bad[16/16],
    [gate tol 1.0 (local)], [91.18 / 89.72], [1.79], [0/16], [5.55], bad[9/16],
    [recip EMA (local)], [91.07 / 89.36], [4.04], [2/16], [4.13], [1/16],
    [gate tol 0.3 (bal)], [89.99 / 89.72], [2.42], [2/16], [1.59], [5/16],
    [*W2 = recip L3 G2 (bal)*], [90.00 / 89.72], good[2.56], good[0/16], good[2.34], good[0/16],
  )
  #note[ρ = 1.25; $c_A^*$ = max trên node của cận trên holdout (mean + 2.776 SE); "node có lợi" = lợi ích holdout có ý nghĩa. q của biến thể bal đo trên val cân bằng nên không so trực tiếp với local.]
  #v(0.3em)
  #grid(columns: (1fr, 1fr), column-gutter: 0.6cm,
    card(fill: rgb("#e9f5ec"))[
      *W2 là biến thể duy nhất có 0/16 node lệch có lợi ở cả hai metric với mọi ρ ∈ [1.25, 3]* (audit ρ = 1: 1/16)\
      ($c_A^* times 100$ ≤ 2.72 cho q, ≤ 2.44 cho audit).\
      Chi phí hằng cần để all-D là Nash giảm từ ≈ 23–38 pp (uniform) xuống ≈ 2.3–2.6 pp.
    ],
    card[
      - Mọi node W2 chọn $s = 2$: tấn công còn sót là loại *dưới ngưỡng phát hiện*, gây hại nhỏ.
      - Không mất chất lượng khi sạch: audit 89.72 vs 89.73.
      - Gate tol 1.0 rẻ theo q nhưng 9/16 node vẫn có lợi theo audit → thước đo quyết định kết luận.
    ],
  )
]

// ───────────────────────────── Wrap-up
#slide[Tổng kết phát hiện][
  #tbl(columns: (auto, 1fr), align: (center, left),
    [*B1*], [DFL hợp tác có giá trị lớn (+12.8 pp audit, node yếu +23 pp) → có cái để tấn công và có cái để mất.],
    [*B2*], [Thiệt hại có ngưỡng (λ ≈ 3–4) rồi lồi; lan theo hop; attacker tự chịu phần lớn thiệt hại.],
    [*B3*], [Best response *nhảy bậc* 0 → 6–8 tại $c^* approx 10^(-3)$; ρ lớn (tự hại nặng) răn đe tự nhiên.],
    [*B4*], [Uniform: Nash có 1–2 attacker khi $c$ nhỏ, kéo $q$ cả mạng giảm 4.9–8.6 pp.],
    [*B5*], [Mọi defense chặn λ lớn; chỉ gate/median/trimmed giữ all-D Nash theo q; clip/BALANCE thì không.],
    [*B6*], [Theo audit, gate/recip với val cục bộ → *tất cả node tấn công* (xác nhận holdout 11–14/16): lỗi lệch thước đo.],
    [*B7*], [Val cân bằng + tol 0.3 + trừng phạt có ân xá: răn đe (own drop 2–3 pp) mà không mất chất lượng node yếu.],
    [*all-D*], [W2: 0/16 node có lợi ích lệch có ý nghĩa ở cả hai metric (ρ ≥ 1.25); $c_A^*$ nhỏ hơn uniform 9–16×.],
  )
  #v(0.3em)
  #card[*Thông điệp chính:* lọc ≠ răn đe. Răn đe đến từ (1) bộ lọc nhìn *cùng thước đo* với utility và (2) *trừng phạt có đi có lại* làm attacker mất lợi ích hợp tác — có ân xá để không phá hợp tác trung thực.]
]

#slide[Hạn chế và bước tiếp theo][
  #grid(columns: (1fr, 1fr), column-gutter: 0.6cm,
    [
      *Hạn chế hiện tại*
      - Một topology (lattice 4×4), một dataset (MNIST), một mức non-IID (α = 1)
      - Lớp chính sách tấn công tĩnh (λ cố định từ vòng 10); chưa có attacker thích ứng/biết W
      - W2 mới kiểm tra *lệch đơn phương* từ all-D; chưa chạy BR N người (B4) theo audit
      - $c_A^*$ của W2 vẫn \> 0: cần giả định chi phí tấn công ≥ ≈ 2.6 pp
      - Val cân bằng là giả định "dịch vụ kiểm định riêng" của mỗi node
      - Chỉ 5 + 5 seed; tham số W2 (L = 3, G = 2, tol 0.3) chọn trên search seeds
    ],
    [
      *Bước tiếp theo*
      + B4-style BR N người cho W2 theo cả q và audit, kiểm tra holdout
      + Attacker thích ứng: λ thay đổi theo thời gian, nhắm vòng cuối (endgame), tấn công λ ≈ 2 kéo dài
      + Độ nhạy: topology small-world / scale-free, α ∈ {0.1, 0.5, ∞}, tỷ lệ attacker, IID
      + Loại tấn công khác: label flipping, backdoor
      + Chứng minh điều kiện: all-D Nash khi $c_A >= max_i G_i$ với cận cho G dưới W2
      + Khóa protocol rồi mới đánh giá trên official test
    ],
  )
]
