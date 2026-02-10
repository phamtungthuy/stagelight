# Stage Light Effect Prediction

## Bài toán

**Input:** Bài nhạc (mp3) → Beat detection tự động → JSON với timestamps  
**Output:** Gán hiệu ứng ánh sáng (`groupLights`) cho từng beat

- ✅ **When** — Xác định thời điểm beat (đã giải quyết, dùng SuperFlux + CGD smoothing)
- 🔲 **What** — Dự đoán hiệu ứng gì cho mỗi beat (chưa giải quyết)

## Cấu trúc JSON

```json
{
  "beats": [
    {
      "time": 0.639977336,
      "duration": 8.362675664,
      "groupLights": [
        {
          "groupLightKey": 0,
          "MotionEffect": 0,
          "ColorEffect": 0,
          "IntensityEffect": 0,
          "VfxEffect": 1
        },
        {
          "groupLightKey": 2,
          "MotionEffect": 4,
          "ColorEffect": 1,
          "IntensityEffect": 3,
          "VfxEffect": 0
        }
      ]
    }
  ]
}
```

## Phân loại groupLightKey

| Key     | Nhóm               | Mô tả                                 |
| ------- | ------------------ | ------------------------------------- |
| 0, 1    | VFX_GROUP          | Hiệu ứng đặc biệt (pháo hoa, khói...) |
| 2, 3    | SINGLE_LIGHT_GROUP | Đèn đơn, chuyển động linh hoạt        |
| 4, 5, 6 | MULTI_LIGHT_GROUP  | Nhóm nhiều đèn, pattern đồng bộ       |

## Enum hiệu ứng

### MotionEffectType

| Giá trị | Tên           | Mô tả              |
| ------- | ------------- | ------------------ |
| 0       | None          | Tắt                |
| 1       | LaserCone     | Chùm tia hình nón  |
| 2       | LaserFan      | Chùm tia hình quạt |
| 3       | Wave          | Chuyển động sóng   |
| 4       | Rotate        | Xoay giữa 2 góc    |
| 5       | Circle_Rotate | Xoay tròn          |
| 6       | PingPong      | Đung đưa qua lại   |

### ColorEffectType

| Giá trị | Tên           | Mô tả                            |
| ------- | ------------- | -------------------------------- |
| 0       | None          | Tắt                              |
| 1       | StaticColor   | Màu cố định đã cấu hình          |
| 2       | RandomPerBeam | Mỗi beam màu ngẫu nhiên riêng    |
| 3       | PingPongColor | Màu chuyển qua lại giữa các beam |

### IntensityEffectType

| Giá trị | Tên               | Mô tả                                     |
| ------- | ----------------- | ----------------------------------------- |
| 0       | None              | Tắt                                       |
| 1       | SpectrumBased     | Độ sáng theo phổ âm thanh (bass/mid/high) |
| 2       | PingPongIntensity | Độ sáng nhấp nháy qua lại                 |
| 3       | AlternatingBeams  | Beam luân phiên sáng/tối                  |
| 4       | WaveIntensity     | Độ sáng lan tỏa kiểu sóng                 |

### VfxEffectType

| Giá trị | Tên              | Mô tả                        |
| ------- | ---------------- | ---------------------------- |
| 0       | None             | Tắt                          |
| 1       | VFX_Simultaneous | Tất cả VFX bật/tắt cùng lúc  |
| 2       | VFX_Wave         | VFX bật/tắt có delay lan tỏa |

## Ràng buộc

- **VFX_GROUP (key 0, 1):** Chỉ dùng `VfxEffect` (1 hoặc 2), các field khác = 0
- **SINGLE_LIGHT_GROUP (key 2, 3):** Thường dùng Motion 3–6 (Wave, Rotate, Circle_Rotate, PingPong)
- **MULTI_LIGHT_GROUP (key 4, 5, 6):** Thường dùng Motion 1–2 (LaserCone, LaserFan)
- Giá trị `0` ở bất kỳ field nào = tắt hiệu ứng đó
- Mỗi beat có thể có nhiều `groupLights` hoạt động cùng lúc, hoặc rỗng (`[]`)