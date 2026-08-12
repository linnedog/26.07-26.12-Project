import os
import torch

# PyTorch 보안 정책 우회 패치
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
  kwargs["weights_only"] = False
  return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO

# 모델 로드
model = YOLO(r"C:\Users\KTR\Desktop\이호성 작업\Ver2.0_test\best.pt")

print("--- [확인] 모델에 등록된 클래스 목록 ---")
print(model.names)

# 폴더 경로
folder_path = r"C:\Users\KTR\Desktop\이호성 작업\Ver2.0_test"
classes_txt_path = os.path.join(folder_path, "classes.txt")

# classes.txt 파일 생성 (로보플로우가 인식하는 형식)
with open(classes_txt_path, "w", encoding="utf-8") as f:
  for i in range(len(model.names)):
    f.write(f"{model.names[i]}\n")

print(
    f"\n[완료] Ver2.0_test 폴더에 classes.txt 파일이 생성되었습니다!"
    " 이 파일도 함께 업로드됩니다."
)