# Đề xuất cải tiến ODAM-Train dựa trên kết quả thực nghiệm

**Ngày tổng hợp:** 2026-07-28
**Bối cảnh:** Faster R-CNN, RCNN-ODAM, ODAM-NMS và SAB-ODAM trên bộ dữ liệu `drill_bit_coco`

---

## 1. Bối cảnh và kết quả thực nghiệm

Kết quả hiện tại cho thấy Faster R-CNN baseline vẫn là mô hình detection mạnh nhất:

| Mô hình | mAP50:95 | mAP50 | mAP75 |
|---|---:|---:|---:|
| Faster R-CNN baseline | **0.3614** | **0.7874** | **0.2487** |
| RCNN-ODAM gốc | 0.2775 | 0.6805 | 0.1586 |
| RCNN-ODAM + ODAM-NMS | 0.2942 | 0.7025 | 0.1735 |
| RCNN-ODAM + schedule + ODAM-NMS | 0.2783 | 0.6811 | 0.1600 |
| SAB-ODAM | 0.2884 | 0.6978 | 0.1768 |
| SAB late-light | 0.2859 | 0.6985 | 0.1599 |

Các quan sát chính:

1. Faster R-CNN baseline vượt tất cả các biến thể ODAM/SAB.
2. ODAM-NMS cải thiện rõ so với ODAM-Train gốc.
3. Schedule ODAM loss đơn thuần không giải quyết được vấn đề.
4. SAB mạnh hơn hoặc phức tạp hơn không đồng nghĩa với detection tốt hơn.
5. Khoảng cách lớn tại mAP75 cho thấy localization chính xác có thể đang bị ảnh hưởng.
6. Threshold sweep chỉ cải thiện hậu xử lý, không chứng minh rằng ODAM/SAB training đã vượt baseline.

Từ đó có thể đưa ra giả thuyết:

> ODAM heatmap vẫn chứa thông tin có ích cho hậu xử lý và phân biệt detection, nhưng gradient từ ODAM loss đang tạo ra negative transfer đối với nhiệm vụ detection chính.

---

## 2. Vấn đề của ODAM-Train hiện tại

ODAM-Train tối ưu đồng thời hai mục tiêu:

\[
L_{\text{total}}
=
L_{\text{det}}
+
\lambda L_{\text{ODAM}}
\]

Trong đó:

- \(L_{\text{det}}\): loss detection của Faster R-CNN.
- \(L_{\text{ODAM}}\): loss dùng để điều chỉnh tính tương đồng hoặc khác biệt giữa các ODAM heatmap.
- \(\lambda\): trọng số cố định hoặc được schedule.

Vấn đề là trọng số nhỏ không đảm bảo ODAM gradient an toàn.

Đặt:

\[
g_d = \nabla_\theta L_{\text{det}}
\]

\[
g_o = \nabla_\theta L_{\text{ODAM}}
\]

Ngay cả khi \(\lambda\) nhỏ, \(g_o\) vẫn có thể:

- Ngược hướng với \(g_d\).
- Có norm quá lớn.
- Tác động vào RPN hoặc box regression.
- Được sinh từ prediction chưa đáng tin cậy.
- Bị chi phối bởi ảnh có quá nhiều proposal hoặc pair.
- Làm giảm localization trong khi cải thiện heatmap discrimination.

Do đó, chỉ dùng warm-up hoặc schedule theo epoch là chưa đủ.

---

# 3. Hướng cải tiến đề xuất: Detection-Preserving Adaptive ODAM

Tên tạm thời:

- **DP-ODAM**
- **Safe-ODAM Train**
- **Primary-Task-Preserving ODAM-Train**

Mục tiêu trung tâm:

> ODAM chỉ được phép cập nhật detector khi gradient của nó không gây hại đáng kể cho nhiệm vụ detection chính.

Kiến trúc đề xuất gồm bốn thành phần:

1. Asymmetric gradient protection.
2. Adaptive ODAM weight theo gradient magnitude.
3. Reliability-aware pair mining.
4. Branch-selective ODAM optimization.

Phiên bản nâng cao có thể bổ sung EMA teacher.

---

## 4. Thành phần 1: Asymmetric Gradient Protection

### 4.1. Đo xung đột gradient

Tính cosine similarity giữa gradient detection và gradient ODAM:

\[
c_t =
\frac{
g_d^\top g_o
}{
\lVert g_d\rVert
\lVert g_o\rVert
+
\epsilon
}
\]

Ý nghĩa:

- \(c_t > 0\): hai gradient tương đối hỗ trợ nhau.
- \(c_t \approx 0\): gần trực giao.
- \(c_t < 0\): gradient ODAM xung đột với detection.

### 4.2. Hard gradient gate

Phiên bản đơn giản nhất:

\[
g_{\text{final}}
=
\begin{cases}
g_d + \lambda g_o, & c_t \geq \tau_c \\
g_d, & c_t < \tau_c
\end{cases}
\]

Các ngưỡng nên thử:

\[
\tau_c \in \{-0.1, 0, 0.1\}
\]

Ưu điểm:

- Dễ triển khai.
- Dễ ablation.
- Fail-safe: khi không chắc chắn thì chỉ tối ưu detection.

### 4.3. Asymmetric gradient projection

Khi \(c_t < 0\), chỉ chỉnh sửa gradient ODAM:

\[
g_o^{\perp}
=
g_o
-
\frac{
g_o^\top g_d
}{
\lVert g_d\rVert^2+\epsilon
}
g_d
\]

Sau đó:

\[
g_{\text{final}}
=
g_d
+
\lambda_{\text{conflict}}
g_o^{\perp}
\]

với:

\[
\lambda_{\text{conflict}} < \lambda
\]

Khác với gradient surgery đối xứng, detection gradient được giữ nguyên vì detection là primary task.

### 4.4. Chính sách fail-closed

Có thể tắt ODAM hoàn toàn trong batch nếu:

- Cosine similarity quá âm.
- ODAM gradient không hữu hạn.
- Gradient norm vượt giới hạn.
- Không đủ reliable pair.
- Validation detection giảm liên tục.
- Heatmap không đạt điều kiện stability tối thiểu.

---

## 5. Thành phần 2: Adaptive ODAM Weight

Một gradient có thể cùng hướng với detection nhưng quá lớn. Vì vậy cần giới hạn tỷ lệ norm.

Đề xuất:

\[
\lambda_t
=
\lambda_{\max}
q_t
\min
\left(
1,
\rho
\frac{
\lVert g_d\rVert
}{
\lVert g_o\rVert+\epsilon
}
\right)
\]

Trong đó:

- \(\lambda_{\max}\): ODAM weight tối đa.
- \(q_t\): độ tin cậy tổng hợp của các ODAM pair trong batch.
- \(\rho\): tỷ lệ norm ODAM tối đa so với detection.
- \(t\): training step.

Các giá trị nên thử:

\[
\rho \in \{0.05, 0.1, 0.2\}
\]

Ví dụ với \(\rho=0.1\), gradient ODAM không được lớn hơn khoảng 10% gradient detection.

### 5.1. Biến thể theo cosine

Có thể thêm cosine vào trọng số:

\[
\lambda_t
=
\lambda_{\max}
q_t
\max(0,c_t)^\gamma
\min
\left(
1,
\rho
\frac{
\lVert g_d\rVert
}{
\lVert g_o\rVert+\epsilon
}
\right)
\]

Trong đó \(\gamma\) điều khiển độ mạnh của gate mềm.

### 5.2. Chỉ số cần log

Mỗi iteration hoặc mỗi \(N\) step nên log:

- \(\lVert g_d\rVert\)
- \(\lVert g_o\rVert\)
- \(c_t\)
- \(\lambda_t\)
- Tỷ lệ batch bị gate.
- Tỷ lệ batch bị projection.
- Số reliable pair.
- Số positive/negative pair.
- Detection loss trước và sau khi bật ODAM.
- Validation mAP theo từng phase.

---

## 6. Thành phần 3: Reliability-Aware ODAM Pair Mining

ODAM heatmap phụ thuộc vào prediction hiện tại. Prediction yếu có thể tạo explanation không đáng tin.

### 6.1. Reliable prediction

Chỉ sử dụng một prediction nếu:

\[
\operatorname{IoU}(b_i,b_i^{gt})
\geq
\tau_{\text{IoU}}
\]

\[
p_i(y^{gt})
\geq
\tau_{\text{cls}}
\]

và prediction đúng class.

Giá trị khởi đầu:

- \(\tau_{\text{IoU}} = 0.5\)
- \(\tau_{\text{cls}} = 0.5\)

Có thể dùng curriculum:

- Giai đoạn đầu của ODAM phase: IoU ≥ 0.7.
- Giai đoạn sau: giảm xuống 0.5.
- Prediction không match GT không được tạo positive pair.

### 6.2. Pair reliability score

Với pair \((i,j)\):

\[
q_{ij}
=
\sqrt{
\operatorname{IoU}_i
\operatorname{IoU}_j
p_i
p_j
}
\cdot
s_{ij}^{\text{stable}}
\]

Trong đó:

- \(p_i,p_j\): confidence đúng class.
- \(s_{ij}^{\text{stable}}\): độ ổn định heatmap qua augmentation nhẹ hoặc qua thời gian.

ODAM loss có trọng số:

\[
L_{\text{ODAM}}
=
\frac{
\sum_{(i,j)}
q_{ij}
L_{ij}
}{
\sum_{(i,j)}
q_{ij}
+
\epsilon
}
\]

### 6.3. Same-object pair

Ưu tiên:

1. Hai proposal khác nhau cùng match một GT.
2. Cùng object dưới hai augmentation.
3. Student và EMA teacher của cùng object.
4. Prediction tại hai thời điểm gần nhau khi matching ổn định.

Không sử dụng:

- Self-pair.
- Prediction không match GT.
- Pair khác ảnh nếu không có object identity đáng tin cậy.

### 6.4. Different-object pair

Không cần dùng tất cả negative pair.

Chỉ giữ hard negative khi:

- Hai object gần nhau.
- Bounding box chồng lấn.
- Cùng class.
- Dễ nhầm class.
- Có nguy cơ suppress nhầm trong NMS.
- Heatmap hiện tại quá giống nhau.

Định nghĩa mẫu:

\[
\mathcal P_{\text{negative}}
=
\left\{
(i,j):
GT_i \neq GT_j,\;
\operatorname{IoU}(b_i,b_j)
>
\tau_{\text{overlap}}
\right\}
\]

Đây là hướng **NMS-aware hard-pair mining**, phù hợp với việc ODAM-NMS đang là thay đổi có ích nhất trong kết quả thực nghiệm.

### 6.5. Pair normalization

Loss không nên chỉ chia theo tổng số pair.

Ảnh có nhiều object hoặc proposal có thể tạo số pair tăng theo bình phương và chi phối batch.

Nên normalize theo:

- Số object.
- Số GT được cover.
- Số pair hợp lệ trên từng object.
- Sau đó mới lấy trung bình toàn batch.

---

## 7. Thành phần 4: Branch-Selective ODAM Optimization

Khoảng cách lớn tại mAP75 gợi ý localization đang bị ảnh hưởng.

Do đó không nên để ODAM gradient cập nhật toàn bộ Faster R-CNN.

### 7.1. Các nhánh nên được bảo vệ

Không cho ODAM gradient cập nhật:

- RPN objectness.
- RPN box regression.
- ROI box regression head.
- Các parameter trực tiếp sinh bounding-box offset.

Tức là:

\[
\nabla_{\theta_{\text{RPN}}}
L_{\text{ODAM}}
=
0
\]

\[
\nabla_{\theta_{\text{box}}}
L_{\text{ODAM}}
=
0
\]

### 7.2. Các nhánh ODAM có thể cập nhật

Giai đoạn đầu:

- ROI classification head.
- Projection head riêng cho ODAM.

Sau khi an toàn:

- ROI feature representation.
- Backbone stage cuối với gradient cap nhỏ.

Không nên mở ODAM gradient xuống toàn backbone ngay từ đầu.

### 7.3. ODAM adapter riêng

Có thể thêm một lightweight adapter:

\[
z_{\text{odam}}
=
A(f_{\text{roi}})
\]

Trong đó:

- \(f_{\text{roi}}\): ROI feature.
- \(A\): ODAM adapter nhỏ.
- ODAM loss chủ yếu cập nhật \(A\).
- Detection head tiếp tục nhận feature gốc hoặc residual nhỏ.

Ví dụ:

\[
f'_{\text{roi}}
=
f_{\text{roi}}
+
\alpha A(f_{\text{roi}})
\]

với \(\alpha\) nhỏ và learnable hoặc được cap.

Cách này cô lập auxiliary objective tốt hơn so với cập nhật trực tiếp toàn detector.

---

## 8. Schedule đề xuất

Schedule đơn thuần đã không đủ trong kết quả hiện tại. Tuy nhiên schedule vẫn hữu ích khi kết hợp gradient protection.

### Phase 1: Detection warm-up

Khoảng 40–50% tổng số epoch:

\[
L = L_{\text{det}}
\]

Mục tiêu:

- Xây dựng detector đủ ổn định.
- Tạo prediction và ODAM heatmap đáng tin cậy.
- Không để explanation loss can thiệp quá sớm.

### Phase 2: Safe ODAM

Khoảng 30–40% epoch:

\[
L
=
L_{\text{det}}
+
\lambda_t L_{\text{ODAM}}
\]

Bắt buộc kèm:

- Reliable pair gating.
- Gradient conflict gate hoặc projection.
- Gradient norm cap.
- Branch isolation.

### Phase 3: Detection recovery

Khoảng 5–10 epoch cuối:

\[
L = L_{\text{det}}
\]

Mục tiêu:

- Phục hồi localization.
- Phục hồi score calibration.
- Không để auxiliary loss ảnh hưởng checkpoint cuối.

### Validation-aware gate

Nếu validation mAP50:95 giảm quá tolerance:

\[
\Delta \operatorname{mAP}_{50:95}
<
-\varepsilon
\]

với ví dụ:

\[
\varepsilon = 0.005
\]

trong hai lần validation liên tiếp, thực hiện một trong các hành động:

1. Giảm \(\lambda_{\max}\).
2. Tăng \(\tau_c\).
3. Tăng độ khắt khe của reliable pair.
4. Tạm tắt ODAM.
5. Chuyển sớm sang recovery phase.

---

## 9. Phiên bản nâng cao: EMA Teacher ODAM

Teacher là trung bình lũy thừa của student:

\[
\theta_T
\leftarrow
\mu\theta_T
+
(1-\mu)\theta_S
\]

Teacher tạo:

- Prediction ổn định hơn.
- GT matching ổn định hơn.
- ODAM heatmap target.
- Pair reliability score.

Student học consistency:

\[
L_{\text{teacher}}
=
D
\left(
T(H_T),
H_S
\right)
\]

Trong đó:

- \(H_T\): heatmap teacher.
- \(H_S\): heatmap student.
- \(T\): biến đổi heatmap tương ứng augmentation.
- Teacher được stop-gradient.

Không nên triển khai teacher ngay từ đầu. Chỉ nên thêm sau khi DP-ODAM cơ bản đã bảo vệ được detection mAP.

---

# 10. Công thức tổng quát của DP-ODAM

Đề xuất cuối cùng:

\[
g_{\text{final}}
=
g_{\text{det}}
+
\lambda_t
q_t
\mathcal P_{\text{safe}}
\left(
M_{\text{branch}}
\odot
g_{\text{ODAM}}
\right)
\]

Trong đó:

- \(g_{\text{det}}\): gradient detection.
- \(g_{\text{ODAM}}\): gradient explanation.
- \(q_t\): pair reliability.
- \(\lambda_t\): adaptive gradient-ratio weight.
- \(M_{\text{branch}}\): mask nhánh được phép nhận ODAM gradient.
- \(\mathcal P_{\text{safe}}\): hard gate hoặc asymmetric projection.
- \(\odot\): masking theo parameter group.

---

# 11. Pseudocode huấn luyện

```python
for images, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)

    # 1. Detection forward
    det_outputs = model(images, targets)
    loss_det = compute_detection_loss(det_outputs)

    # 2. Tạo reliable predictions/pairs
    matches = match_predictions_to_ground_truth(
        det_outputs,
        targets,
        iou_threshold=tau_iou,
        cls_threshold=tau_cls,
    )

    reliable_pairs = build_reliable_odam_pairs(
        matches,
        use_hard_negative_mining=True,
        normalize_per_object=True,
    )

    if not odam_phase or len(reliable_pairs) < min_pairs:
        loss_det.backward()
        optimizer.step()
        continue

    # 3. ODAM loss
    loss_odam, pair_quality = compute_weighted_odam_loss(
        model=model,
        predictions=det_outputs,
        pairs=reliable_pairs,
    )

    # 4. Gradient diagnostics
    g_det = autograd_grad(loss_det, protected_parameters, retain_graph=True)
    g_odam = autograd_grad(loss_odam, odam_parameters, retain_graph=True)

    cosine = gradient_cosine(g_det, g_odam)
    det_norm = gradient_norm(g_det)
    odam_norm = gradient_norm(g_odam)

    # 5. Adaptive weight
    lambda_t = compute_adaptive_lambda(
        lambda_max=lambda_max,
        pair_quality=pair_quality,
        det_norm=det_norm,
        odam_norm=odam_norm,
        ratio_cap=rho,
    )

    # 6. Safe ODAM gradient
    if cosine < conflict_threshold:
        if strategy == "gate":
            safe_odam_grad = zeros_like(g_odam)
        else:
            safe_odam_grad = asymmetric_project(g_odam, g_det)
            lambda_t *= conflict_scale
    else:
        safe_odam_grad = g_odam

    # 7. Branch mask
    safe_odam_grad = apply_branch_mask(
        safe_odam_grad,
        allow_roi_classifier=True,
        allow_roi_box_regressor=False,
        allow_rpn=False,
        allow_backbone_last_stage=enable_backbone_odam,
    )

    # 8. Merge gradients
    assign_detection_gradients(g_det)
    add_scaled_odam_gradients(safe_odam_grad, scale=lambda_t)

    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
```

---

# 12. Ma trận thí nghiệm tối thiểu

| Experiment | Gradient protection | Reliable pairs | Branch isolation | ODAM-NMS | Teacher |
|---|---:|---:|---:|---:|---:|
| E0 Faster R-CNN baseline | – | – | – | No | No |
| E1 Baseline + ODAM-NMS | – | – | – | Yes | No |
| E2 Original ODAM | No | No | No | Yes | No |
| E3 Safe-gradient ODAM | Yes | No | No | Yes | No |
| E4 Reliability ODAM | Yes | Yes | No | Yes | No |
| E5 DP-ODAM | Yes | Yes | Yes | Yes | No |
| E6 DP-ODAM + Teacher | Yes | Yes | Yes | Yes | Yes |

### E1: Baseline + ODAM-NMS

Mục đích:

- Tách riêng lợi ích của hậu xử lý.
- Không quy cải thiện ODAM-NMS cho ODAM training.

### E3: Safe-gradient ODAM

Ablation:

- Gate với \(\tau_c=-0.1\).
- Gate với \(\tau_c=0\).
- Gate với \(\tau_c=0.1\).
- Projection bất đối xứng.

### E4: Reliability ODAM

Ablation:

- Confidence-only.
- IoU-only.
- Confidence + IoU.
- Confidence + IoU + heatmap stability.
- Hard negative mining bật/tắt.

### E5: Branch isolation

Ablation:

1. ROI classifier only.
2. ROI classifier + ODAM adapter.
3. ROI classifier + last backbone stage.
4. Full backbone.
5. Có hoặc không box regression isolation.

---

# 13. Chỉ số đánh giá

## 13.1. Detection

Bắt buộc:

- mAP50:95.
- mAP50.
- mAP75.
- AP theo lớp.
- AP medium.
- AP large.
- AR100.
- Precision/recall.
- Số prediction trên mỗi ảnh.
- False positive/false negative theo lớp.

Không dùng mAP small làm kết luận chính vì test split chỉ có một object nhỏ.

## 13.2. Explanation

Nên đánh giá:

- Same-object heatmap similarity.
- Different-object heatmap separation.
- Heatmap stability qua augmentation.
- Pointing game.
- Saliency IoU hoặc energy ratio trong bounding box.
- Tỷ lệ pair vi phạm semantics.
- Tương quan explanation quality với detection correctness.

## 13.3. Optimization

Log:

- Gradient cosine.
- Gradient norm ratio.
- Gate rate.
- Projection rate.
- Adaptive \(\lambda_t\).
- Số reliable pair.
- Pair quality distribution.
- Detection loss drift.
- ODAM loss drift.
- Thời gian và memory overhead.

---

# 14. Tiêu chí thành công

Mục tiêu đầu tiên không nên là vượt baseline ngay.

### Stage 1: Không làm hỏng detection

\[
\Delta mAP_{50:95}
\geq
-0.005
\]

so với Faster R-CNN baseline, đồng thời explanation metric tốt hơn.

### Stage 2: Có lợi ích cục bộ rõ ràng

Ví dụ:

- Giảm false suppression.
- Cải thiện recall cho object gần nhau.
- Cải thiện lớp khó.
- Cải thiện detection trong ảnh có nhiều object chồng lấn.
- Giữ hoặc tăng mAP75.

### Stage 3: Vượt baseline

Chỉ tuyên bố vượt baseline khi:

- Nhiều seed.
- Cùng protocol.
- Cùng split.
- Checkpoint selection công bằng.
- Có confidence interval hoặc kiểm định phù hợp.
- Không chỉ cải thiện threshold hậu xử lý.

---

# 15. Thứ tự triển khai khuyến nghị

## Bước 1: Instrumentation

Trước khi đổi loss:

- Log cosine giữa detection và ODAM gradient.
- Log norm ratio.
- Log theo parameter group.
- Phân tích conflict theo epoch, class và loại pair.

Mục tiêu là xác nhận negative transfer.

## Bước 2: Hard gradient gate

Triển khai phiên bản đơn giản nhất:

\[
c_t < 0
\Rightarrow
\text{không dùng ODAM gradient}
\]

Đây là bước có chi phí thấp và giá trị chẩn đoán cao.

## Bước 3: Branch isolation

Tắt ODAM gradient với:

- RPN.
- ROI box regression.

Chỉ cho cập nhật ROI classifier hoặc adapter.

## Bước 4: Adaptive norm cap

Giới hạn ODAM gradient ở mức:

- 5%.
- 10%.
- 20%.

so với detection gradient.

## Bước 5: Reliable pair mining

Thêm:

- GT matching.
- IoU/confidence threshold.
- Hard negative mining.
- Per-object normalization.

## Bước 6: Recovery phase

Tắt ODAM ở cuối training để detector phục hồi.

## Bước 7: EMA teacher

Chỉ triển khai nếu DP-ODAM cơ bản đã giữ được detection performance.

---

# 16. Đóng góp nghiên cứu tiềm năng

Một bài nghiên cứu có thể tập trung vào ba đóng góp:

## Contribution 1: Asymmetric gradient protection

Đề xuất cơ chế bảo vệ primary detection task trong explainability-guided object detection training.

## Contribution 2: Reliability-aware ODAM learning

Chỉ sử dụng explanation được sinh từ prediction đủ tin cậy và weighting theo chất lượng pair.

## Contribution 3: Branch-selective explanation optimization

Cô lập tác động của explanation loss khỏi localization-sensitive branches.

Tên bài báo tạm thời:

> **Detection-Preserving ODAM Training via Reliability-Aware Pair Mining and Asymmetric Gradient Protection**

Hoặc:

> **Safe Explanation-Guided Training for Two-Stage Object Detectors**

---

# 17. Kết luận

Kết quả hiện tại không cho thấy nên tiếp tục tăng độ phức tạp của SAB loss theo hướng cũ.

Dữ liệu thực nghiệm lại chỉ ra hai tín hiệu:

1. ODAM-NMS có ích.
2. ODAM/SAB training đang làm suy giảm detection.

Vì vậy, hướng hợp lý nhất là xây dựng một ODAM-Train mới với nguyên tắc:

> Explanation loss là auxiliary objective và không được phép gây hại cho detection objective.

DP-ODAM nên kết hợp:

- Gradient conflict gate hoặc asymmetric projection.
- Adaptive ODAM weight theo gradient norm.
- Reliable pair selection.
- NMS-aware hard negative mining.
- Branch isolation.
- Detection recovery phase.
- EMA teacher ở phiên bản nâng cao.

Đây là hướng vừa bám sát kết quả thực nghiệm hiện tại, vừa có khả năng hình thành một đóng góp nghiên cứu có tính mới và có thể kiểm chứng bằng ablation rõ ràng.
