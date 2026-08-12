import os
import torch

# PyTorch 2.6+ 보안 정책으로 인한 UnpicklingError 원천 차단 (신뢰할 수 있는 본인 모델이므로 해제)
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
  kwargs["weights_only"] = False
  return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO

# 1. 모델 로드
model = YOLO(r"C:\Users\KTR\Desktop\이호성 작업\Ver2.0_test\best.pt")

# 2. 이미지 폴더 경로 및 라벨링 생성 로직
img_dir = r"C:\Users\KTR\Desktop\이호성 작업\Ver2.0_test"
valid_exts = (".jpg", ".jpeg", ".png")
images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]

print(f"총 {len(images)}장의 이미지에 대한 자동 라벨링을 시작합니다...")

count = 0
for img_name in images:
  img_path = os.path.join(img_dir, img_name)

  # 신뢰도 0.2로 낮춰서 추론
  results = model.predict(img_path, conf=0.2, verbose=False)

  base_name = os.path.splitext(img_name)[0]
  txt_path = os.path.join(img_dir, f"{base_name}.txt")

  with open(txt_path, "w", encoding="utf-8") as f:
    for r in results:
      for box in r.boxes:
        cls = int(box.cls[0])
        xywhn = box.xywhn[0].tolist()
        f.write(f"{cls} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")

  count += 1

print(
    f"작업 완료! 총 {count}개 이미지에 대한 .txt 라벨 파일이 폴더에"
    " 생성되었습니다."
)