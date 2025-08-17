# from_scratch_LLM

### 한 파일로 구현한 Decoder-Only Transformer(GPT) 튜토리얼.
바이그램 → 셀프어텐션 → 멀티헤드 → FFN → 잔차 → 레이어정규화 → 드롭아웃까지, 왜 필요한지와 어떻게 구현하는지를 코드 중심으로 따라갑니다. 모델은 한 문자 단위 Tiny Shakespeare로 학습해 다음 토큰 예측과 텍스트 생성을 시연합니다.

---

### 핵심 아이디어
언어 모델링: “이전 토큰들”로부터 다음 토큰 확률을 예측하는 자기회귀 LM.
마스킹된(인과적) 셀프 어텐션: 미래를 보지 않도록 하삼각 마스크 + 스케일드 닷프로덕트(Q·K/√d).
멀티헤드 어텐션: 여러 관점의 주의집중을 병렬로 학습해 표현력을 확장.
FFN(MLP): 토큰별 비선형 변환으로 “통신 뒤 계산” 수행.
잔차 연결 & Pre-LayerNorm: 깊은 네트워크 최적화(기울기 흐름 안정화).
드롭아웃: 정규화로 과적합 방지.
샘플링(generate): 소프트맥스 분포에서 자기회귀로 토큰을 반복 샘플링.

### 데이터 & 토크나이저
Tiny Shakespeare (~1MB) 문자 수준, V≈65.
encoder/decoder로 문자↔정수 매핑.
train/val=90/10, **get_batch(B,T)**로 무작위 청크 샘플링(컨텍스트=block_size).
구현 구성요소(Why & How)
Token/Positional Embedding: 정체성과 순서를 합산해 입력 표현 구성.
Causal Mask: tril로 미래 차단 → 언어모델의 순차성 보장.
Scaled Dot-Product Attention: QKᵀ/√d 후 softmax → 데이터 의존 가중 합.
Multi-Head: 독립 head들의 출력을 concat해 다양한 패턴 포착.
FFN (GELU/ReLU): 토큰별 비선형 변환(논문 대비 4× 확장 채널 권장).
Residual + Projection: 잔차 경로로 안정적 최적화, 채널 정합.
Pre-LayerNorm: 각 변환 이전에 정규화해 학습 안정화.
Dropout(0.2): 확장된 설정에서 과적합 완화.

### 한계와 확장
본 코드는 원리 학습용 소규모 문자 LM.
ChatGPT 수준을 위해서는 대규모 토큰/파라미터 + 정렬(Alignment) 단계 필요
(① 지도 미세조정(SFT) ② 보상모델(RM) ③ PPO).

### 학습 포인트
행렬 연산으로 어텐션을 벡터화(하삼각 마스킹·masked_fill(-inf)·softmax).
배치/컨텍스트 샘플링으로 효율적 학습.
검증 손실 추정 루틴으로 과적합 관찰 및 하이퍼 파인튜닝.

### 참고
Transformer: Attention Is All You Need (2017)
교육용 구현 철학: 간결한 코드로 본질 파악 → 필요 시 NanoGPT 스타일로 확장
Goal: 이 저장소를 통해 GPT의 작동 원리를 코드로 끝까지 따라가고,
직접 작은 언어 모델을 학습·생성까지 해보는 것.

lecture : https://youtu.be/kCc8FmEb1nY?si=W8DmM_bUuhxWytAg
