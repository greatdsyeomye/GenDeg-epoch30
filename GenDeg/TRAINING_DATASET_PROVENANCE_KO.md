# 복합 열화 모델 학습 데이터 확인

확인일: 2026-10-20

## 핵심 결론

- **TSCIR**은 CDD-11로 학습한 모델이 아니라 자체 **LCDD**로 학습한 복원 모델이다.
- 공개된 TSCIR 저장소에는 현재 **LCDD 생성 코드만 있고 TSCIR 복원 네트워크/가중치는 없다.** 따라서 동일 BSDS 7장에 LCDD 열화를 생성하는 것은 가능하지만, TSCIR 복원 inference는 불가능하다.
- **OneRestore**의 대표 체크포인트 `onerestore_cdd-11.tar`와 embedder는 **CDD-11**로 학습됐다.
- **ClearNight**는 CDD-11이 아니라 **AllWeatherNight**로 학습한다.
- 공식 **GenDeg** 사전학습 가중치는 CDD-11이 아니라 **GenIRData**를 사용한다. 이 저장소에서 추가로 학습한 체크포인트들은 설정에 따라 CDD-11 단독, CDD-11+AllWeatherNight, 또는 5개 데이터 혼합이다.

## 모델별 학습 데이터

| 모델 | 학습 데이터 | 현재 로컬 상태 | CDD-11 사용 여부 |
|---|---|---|---|
| OneRestore | CDD-11 (11개 열화 유형) | 데이터, 코드, 공식 가중치 있음; inference 완료 | 예 |
| CURE | CCDD-11; OneRestore baseline/encoder로 초기화 | 코드/데이터/가중치 있음 | CDD-11의 controllable 확장판 |
| TSCIR | LCDD: rain-haze 8,250 + snow-rain 9,001 + snow-haze 9,001 = train 26,252쌍 | 생성 코드와 재현 데이터 있음; 복원 코드/가중치 없음 | 아니오 |
| ClearNight | AllWeatherNight train 8,000쌍 | 데이터와 train/test 코드 있음; 공식 복원 가중치 없음 | 아니오 |
| L2RIRNet | 저조도-비 합성/실사 LLR 데이터 | train 8,368쌍, test 합성 800쌍, 실사 input 42장; inference 완료 | 아니오 |
| LVRNet | AFO로부터 만든 Low-Vis AFO (저조도+안개) | 코드/가중치 있음; 데이터셋 본체는 로컬에 없음 | 아니오 |
| Heavy Rain Removal | 합성 Outdoor-Rain | train 9,000쌍 로컬 보유; inference 완료 | 아니오 |
| Defusion | Rain1400, Outdoor-Rain, LHP, RESIDE, Dense-Haze, Snow100K, RealSnow, RainDrop, RainDS, SIDD, GoPro, RealBlur, DPDD 및 WED 혼합 열화 | 논문만 사용 가능; 공식 저장소는 `Code coming soon` | 아니오 |
| 공식 vanilla GenDeg | GenIRData | 공식 가중치 inference 완료 | 아니오 |
| 이 저장소의 custom GenDeg | 체크포인트별로 CDD-11 / CDD-11+AllWeatherNight / CDD-11+AllWeatherNight+LCDD+WeatherBench+LLR | 로컬 설정과 가중치에 따라 다름 | 체크포인트별 상이 |
| WeatherWeaver | 자체 물리 시뮬레이션, 생성 편집, auto-labeled real video 데이터 | 공개 저장소는 데이터/데모 중심; 복원 inference 코드/가중치 없음 | 아니오 |

## OneRestore / CDD-11 세부사항

로컬 `data/cdd11`에서 직접 센 개수:

- train: clean scene 1,183장, 각 열화 category도 1,183장
- test: clean scene 200장, 각 열화 category도 200장
- 열화 category: `low`, `haze`, `rain`, `snow`, `low_haze`, `low_rain`, `low_snow`, `haze_rain`, `haze_snow`, `low_haze_rain`, `low_haze_snow`
- 따라서 열화 train pair는 1,183 × 11 = **13,013쌍**, test는 200 × 11 = **2,200쌍**이다.
- 논문 학습에서는 256×256 crop, stride 200과 회전을 적용해 약 312k patch pair를 만든다.

CDD-11 생성에는 clear image 외에 다음이 필요하다.

- MegaDepth depth map: haze
- LIME illumination/light map: low-light
- RainStreakGen rain mask: rain
- Snow100K snow mask: snow

공식 `syn_data.py`의 실제 적용 순서는 `low → rain 또는 snow → haze`이다. 따라서 폴더명 `haze_rain`은 열화 집합 이름이고, 픽셀 합성 순서는 코드상 `rain → haze`이다. CDD-11에는 `low+rain+snow` 또는 `rain+snow`가 없으며 rain/snow를 상호 배타적으로 다룬다.

공식 OneRestore가 함께 배포하는 단일 열화 전용 체크포인트는 별도 데이터로 학습됐다.

- `onerestore_lol.tar`: LOL
- `onerestore_reside_ots.tar`: RESIDE-OTS
- `onerestore_rain1200.tar`: Rain1200
- `onerestore_snow100k.tar`: Snow100K-L

현재 BSDS 복합 열화 복원 비교에 사용한 것은 이 단일 열화 체크포인트들이 아니라 **`onerestore_cdd-11.tar`**이다.

## TSCIR 생성 코드를 CDD 입력으로 바꿀 수 있는 범위

가능하다. 다만 정확히는 다음과 같다.

1. CDD-11의 `clear`를 Groundtruth 입력으로 사용한다.
2. 각 clear image에 대응하는 depth를 Monodepth2로 만든다.
3. `rain → haze`를 순서대로 렌더링한다.

CDD-11의 이미 열화된 `haze_rain` 폴더만 입력 경로로 넣는 것은 올바른 재생성이 아니다. haze 렌더러가 depth를 요구하기 때문이다. `snow→rain`이나 `snow→haze`는 CDD의 snow 이미지를 중간 입력으로 사용할 수 있지만, TSCIR/LCDD 생성 코드에는 low-light 생성기가 없으므로 `low→rain`, `low→haze`는 만들 수 없다. 그 두 조합은 OneRestore의 CDD 생성기를 사용하는 편이 맞다.

이번에 동일 BSDS 7장과 seed 2026으로 `rain→haze`를 실제 생성했다.

- 결과: `results/other_models/bsds7_tscir_lcdd/rain__to__haze/degraded/`
- rain 중간 결과: `results/other_models/bsds7_tscir_lcdd/rain__to__haze/rain_intermediate/`
- 파라미터 기록: `results/other_models/bsds7_tscir_lcdd/rain__to__haze/metadata.json`
- GenDeg 직접 비교: `results/other_models/bsds7_tscir_lcdd/rain_to_haze_same_input_comparison.jpg`

## Train/test 분리

열화가 자연스러운지만 눈으로 보는 audit에는 train 샘플을 보는 것이 문제없다. 하지만 복원 성능 비교에는 train 이미지를 쓰면 안 된다. scene 단위로 분리된 test set을 유지해야 하며, 같은 clean scene에서 파생된 다른 열화 버전도 train/test를 가로질러 섞지 않는 것이 중요하다.

실제 train split 샘플 비교 이미지는 다음에 저장했다.

- `results/other_models/training_dataset_actual_train_audit.jpg`

## 근거가 되는 로컬 파일

- OneRestore 데이터/체크포인트 설명: `third_party/OneRestore/README.md`
- CDD 생성 순서와 수식 구현: `third_party/OneRestore/syn_data/syn_data.py`
- TSCIR/LCDD 생성 설명: `third_party/TSCIR/README.md`, `third_party/TSCIR/generate_dataset.m`
- ClearNight/AllWeatherNight 학습 코드 설명: `third_party/ClearNight/README.md`
- LVRNet 데이터 설명: `third_party/LVRNet/README.md`
- HeavyRainRemoval 데이터 설명: `third_party/HeavyRainRemoval/README.md`
- 공식 GenDeg 데이터 설명: `README.md`
- 로컬 5-dataset custom GenDeg 설정: `configs/train_composite_residual_55k.yaml`
