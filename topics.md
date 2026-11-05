# Topic
Chống tấn công đầu độc mô hình trong Học Liên Kết Phi tập trung dựa trên mô hình trò chơi chiến lược.

# Overview

Decentralized Federated Learning (DFL) cho phép các agent trao đổi và cập nhật mô hình trực tiếp với nhau theo topology ngang hàng mà không cần server trung tâm. Mặc dù loại bỏ được điểm yếu tập trung, DFL lại đối mặt với thách thức nghiêm trọng hơn về an ninh, đặc biệt là tấn công đầu độc mô hình (model poisoning attack).


Trong DFL, các agent độc hại có thể gửi các cập nhật mô hình bị thao túng cho neighbor của chúng nhằm:


• Làm suy giảm độ chính xác của mô hình trên toàn mạng.

• Chèn backdoor hoặc gây lệch dự đoán theo hướng có lợi cho kẻ tấn công.

• Lan truyền nhanh tác động độc hại do tính chất lan tỏa của giao tiếp ngang hàng.

Khác với Federated Learning tập trung (có server đóng vai trò aggregation và lọc), trong DFL không tồn tại thực thể trung tâm đáng tin cậy để thực hiện các kỹ thuật robust aggregation truyền thống (Krum, Median, Trimmed Mean…). Việc phòng thủ phải được thực hiện một cách phân tán, dựa trên thông tin cục bộ và tương tác với các neighbor.


Đề tài tiếp cận vấn đề từ góc độ Lý thuyết Trò chơi chiến lược. Tương tác giữa các agent trung thực và agent độc hại được mô hình hóa như một trò chơi chiến lược trên mạng, trong đó:


• Agent độc hại chọn chiến lược tấn công (cường độ, thời điểm, đối tượng tấn công…) để tối đa hóa thiệt hại lan tỏa.

• Agent trung thực chọn chiến lược phòng thủ (lọc cập nhật, điều chỉnh trọng số tin cậy, loại trừ neighbor, hoặc thay đổi tần suất giao tiếp…) để giảm thiểu ảnh hưởng của tấn công và duy trì hiệu năng học tập.

Mục tiêu của đề tài là thiết kế các cơ chế phòng thủ phân tán dựa trên phân tích trò chơi chiến lược, giúp hệ thống DFL nâng cao khả năng chống chịu trước tấn công đầu độc mô hình một cách hiệu quả và có cơ sở lý thuyết vững chắc.

# Phase 1

GIAI ĐOẠN 1 (ĐCLV/ ĐACN/ ĐAKTMT):

(1) Nghiên cứu tổng quan


• Khảo sát các hình thức tấn công đầu độc mô hình trong Federated Learning và đặc thù khi chuyển sang môi trường phi tập trung.

• Phân tích các phương pháp phòng thủ hiện có trong DFL (reputation-based filtering, local outlier detection, trusted neighbor selection, blockchain-based approaches…).

• Xác định hạn chế của các phương pháp hiện tại và tiềm năng của tiếp cận dựa trên Lý thuyết Trò chơi chiến lược.

(2) Mô hình hóa bài toán bằng Lý thuyết Trò chơi


• Xác định người chơi: agent trung thực và agent độc hại (có thể mở rộng thành multi-population game).

• Định nghĩa không gian chiến lược:

• Chiến lược của agent độc hại: cường độ nhiễu, loại tấn công (untargeted, targeted, backdoor), lựa chọn neighbor để tấn công, thời điểm tấn công.

• Chiến lược của agent trung thực: ngưỡng lọc cập nhật, trọng số tin cậy với neighbor, quyết định giữ/loại bỏ kết nối, mức độ ảnh hưởng của cập nhật nhận được.

• Xây dựng hàm payoff phản ánh mục tiêu của từng bên (độ chính xác mô hình cục bộ và lan tỏa, chi phí tấn công/phòng thủ, mức độ ảnh hưởng đến neighbor).

• Lựa chọn mô hình trò chơi phù hợp với DFL:

• Network game / Graphical game (chiến lược phụ thuộc vào topology).

• Evolutionary game (sự lan truyền chiến lược tấn công và phòng thủ trên mạng).

• Repeated game hoặc Bayesian game (khi thông tin về tỷ lệ agent độc hại không hoàn hảo).

(3) Thiết kế cơ chế phòng thủ phân tán


• Đề xuất quy tắc cập nhật độ tin cậy (trust/reputation) dựa trên phân tích chiến lược của trò chơi.

• Thiết kế cơ chế lọc hoặc điều chỉnh trọng số cập nhật mô hình nhận từ neighbor.

• Xây dựng chiến lược thích ứng: agent trung thực điều chỉnh hành vi phòng thủ theo thời gian dựa trên quan sát payoff và hành vi của neighbor.

• Đảm bảo tính hoàn toàn phi tập trung: mọi quyết định chỉ dựa trên thông tin cục bộ và trao đổi với neighbor trực tiếp.

# Phase 2
GIAI ĐOẠN 2 (LVTN/ ĐATN): 

(4) Phân tích lý thuyết


• Nghiên cứu sự tồn tại và tính chất của điểm cân bằng (Nash Equilibrium hoặc Evolutionary Stable Strategy) trên mạng.

• Phân tích điều kiện để chiến lược phòng thủ lan rộng và ức chế chiến lược tấn công.

• Đánh giá ảnh hưởng của topology mạng, tỷ lệ agent độc hại và mức độ non-IID đến hiệu quả phòng thủ.

• Phân tích tính ổn định của hệ thống trước các chiến lược tấn công thích ứng.

(5) Thiết kế thuật toán và triển khai


• Xây dựng quy trình DFL tích hợp cơ chế phòng thủ đề xuất.

• Cài đặt trên các framework mô phỏng (Flower, FedML, hoặc PyTorch kết hợp NetworkX để mô phỏng topology và giao tiếp ngang hàng).

• Thiết kế các kịch bản tấn công:

• Tỷ lệ agent độc hại khác nhau.

• Các loại tấn công: gradient manipulation, label flipping, backdoor.

• Topology khác nhau (small-world, scale-free, random geometric…).

• Dữ liệu IID và non-IID.

• Kẻ tấn công có kiến thức đầy đủ hoặc hạn chế về cơ chế phòng thủ.

(6) Đánh giá thực nghiệm và so sánh


• Các chỉ số đánh giá chính:

• Độ chính xác mô hình dưới tấn công (global và local).

• Khả năng hạn chế sự lan truyền của cập nhật độc hại.

• Tỷ lệ phát hiện/loại bỏ ảnh hưởng từ agent độc hại.

• Hiệu năng khi không có tấn công (clean performance).

• Chi phí truyền thông và tính toán thêm.

• So sánh với các baseline: DFL thuần túy, các phương pháp reputation-based, local robust aggregation, và một số defense (BALANCE) gần đây trong DFL.