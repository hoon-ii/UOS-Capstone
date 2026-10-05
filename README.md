# UOS Capstone — 합성데이터 생성 (Synthetic Data Generation)

서울시립대학교 캡스톤 강의 자료입니다.
딥러닝 생성모형으로 정형데이터를 합성하고, 그 데이터를 평가하는 것까지 다룹니다.
이후, 조별로 정의한 문제를 해결할 수 있는 방법을 찾아봅니다. 

참고 저장소: [hoon-ii/SPT](https://github.com/hoon-ii/SPT)

---

## 주차별 내용

| 주차 | 주제 | 다루는 것 |
|:---:|---|---|
| Week 2 | [AutoEncoder와 VAE](week2_VAE.pdf) | 잠재표현, AE의 한계, ELBO, reparameterization, Tabular VAE |
| Week 3 | [GAN과 Diffusion](week3_GAN&DIffusion.pdf) | GAN의 학습과 mode collapse, Diffusion의 기본 질문들 |
| Week 4 | [Latent Space](week4_LatentSpace.pdf) | Tabular 전처리, latent 시각화(PCA·KDE), interpolation |
| Week 5 | [생성과 평가](week5_Evaluate.pdf) | prior 샘플링, β-VAE, 좋은 합성데이터의 다섯 축 |
| Week 6 | [Diffusion](week6_Diffusion.pdf) | Forward/Reverse, 노이즈 스케줄, MNIST 실습, TabDDPM |

각 강의안의 `문제 N` / `RQ N` 은 수업 중 직접 풀고 토론하는 항목입니다.

---

## 실습 코드
기본적인 코드는 week2_VAE.py를 참고하세요. 이 코드를 바탕으로 여러분의 코드를 완성하세요!
| 파일 | 내용 |
|---|---|
| [`week2_VAE.py`](week2_VAE.py) | Tabular VAE 학습 → 생성 → 원본과 비교 → β 스케줄링 |
| [`datasets/`](datasets/) | Wine Quality 로드, 전처리(Quantile·Ordinal)와 역변환 |
| [`utils.py`](utils.py) | 시드 고정 |

데이터: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

```bash
pip install -r requirements.txt
```
