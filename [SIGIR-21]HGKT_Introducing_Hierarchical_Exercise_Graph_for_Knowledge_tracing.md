# [Review] HGKT: Introducing Hierarchical Exercise Graph for Knowledge

## 1. INTRODUCTION

Knowledge Tracing은 컴퓨터를 기반으로 하여 디지털화된 교육이 점차 증가함에 따라 학생의 문제 풀이 sequence를 활용하여 지식상태를 추적하는 과업을 의미한다. 즉, 이전까지 학생이 풀이한 문제와 그 문제의 정오답여부 및 기타 다양한 feature를 활용하여 아직 풀이하지 않은 문제를 풀 수 있는지 여부를 예측하는 것이다. Knowledge Tracing은 Hidden Markov Model을 사용하는 통계적 방식인 Bayesian Knowledge Tracing(BKT)을 시작으로 최근에는 DNNs(Deep Neural Networks)를 활용한 다양한 모델이 연구되고 있고, 성능 또한 매우 빠르게 발전하고 있다. 그러나, 기존 모델들이 공통적으로 가지고 있는 두가지 한계점은 다음과 같다.

1. Information Loss: 문제 풀이 기록과 같은 표면적으로 드러난 데이터만을 활용함으로써 문제의 난이도 혹은 문제에 내재된 다양한 의미들을 고려하지 않아 충분한 정보를 고려하지 않았다.
2. Insufficient Diagnosis problem: 충분하지 않은 정보만으로 지식을 추적함에 따라 학생의 지식상태를 충분히 진단하지 못하여 학생의 학습수준을 정확하게 파악하지 못하였다.

본 연구는 위의 두 가지 한계점을 Hierarchical Graph Structure를 활용하여 보완함으로써 모델의 성능(performance prediction)과 설명력(interpretability)을 높이고자 하였다.  
HGKT의 기본 framework은 Figure 1과 같다.

![Figure 1](https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/blob/main/figure1.png?raw=true)

$$Figure 1$$  
Figure 1의 Training Process에서 학생은 $e_1$과 $e_3$은 맞혔고, $e_2$는 틀렸다. 이때, 'Coordinate Calculation'의 이해도를 동일하게 확인하는 $e_7$과 $e_8$의 문제 풀이 여부를 예측하는 경우, 'Pythagorean Theorem'의 이해도를 확인하는 $e_2$는 틀렸고, $e_7$과 개념적으로 관련있어 $e_2$와 관련된 학생의 지식상태가 $e_7$를 예측하는데 직접적으로 참고가 되고(direct support), $e_3$은 $e_8$과 개념적으로는 연결되지 않고 문제의 스키마(problem schema)만 관련있기 때문에 $e_8$의 문제 풀이 여부를 예측하는데 간접적으로 참고가 된다(indirect support). 여기서 문제의 스키마는 그 문제의 풀이방식을 나타내는 것으로 문제에서 묻고자 하는 수학적 개념과 다르다.  

지금까지 설명된 내용을 바탕으로 본 연구의 contribution을 요약하면 다음과 같다.  
$\bullet$ hierarchical graph를 사용하여 문제 간의 관계를 두 가지 유형(direct support relation, indirect support relation)으로 나타내어 문제 간의 관계를 기존의 Knowledge Tracing 연구보다 정교하게 활용하였다.  
$\bullet$ 각 문제에서 묻는 개념인 knowledge concept뿐만 아니라, 문제의 스키마라는 개념을 새로 도입하여 문제를 더욱 효과적으로 representation함으로써 앞서 언급한 information loss를 줄이고자 하였다.  
$\bullet$ 모델의 architecture에서 두개의 attention mechanism을 사용함으로써 hierarchical graph의 내용을 충분히 반영하고, 각 학생의 지식상태 또한 정확하게 반영하고자 시도하였다.  
$\bullet$ knowledge&schema (K&S) diagnosis matrix를 활용하여 개념과 문제의 스키마의 습득여부(mastery)를 동시에 고려함으로써, 기존 연구의 한계점인 insufficient diagnosis problem을 해결하고자 하였다. knowledge와 schema의 관계와 K&S diagnosis matrix는 Figure 2와 같다.  
![Figure 2](https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/blob/main/figure2.png?raw=true)
$Figure 2$  

## 2. RELATED WORK

Knowledge Tracing과 관련하여 기존의 연구를 간단히 설명하면 다음과 같다.

DKT(Deep Knowledge Tracing): 학생의 문제풀이 데이터에 sequnce가 존재하는 것을 고려하여 RNNs(recurrent neural networks)를 활용한 모델

DKVMN(Dynamic Key-Value-Memory Network): 두 가지 별개의 memory matrix를 활용하여 각 knowledge concept(문제에서 묻는 개념)에 대한 학생의 지식상태와 그 knowledge concept의 지식을 습득한 여부를 RNNS에 반영한 모델

GKT(Graph-based Knowledge Tracing): 각 문제에 대한 학생의 지식상태 잠재변수(hidden knowledge state)를 그래프의 각 노드에 임베딩한 후 그래프 구조를 활용한 모델

위의 모델들은 각 시점에서 SOTA의 성능을 보였지만 학생들이 풀이한 실제 문제의 텍스트 정보와 같은 좀 더 풍부한 정보를 담지 못함으로써 문제 간의 관계를 충분히 고려하지 못한 한계점이 존재한다. 물론, EKT(Exercise Enhanced Knowledge Tracing) 모델이 처음으로 문제의 텍스트를 분석하여 knowledge tracing에 활용하였지만 그 정보를 충분히 사용하지 못함으로써 문제간 관계를 고려하지 못하였다.  

## 3. PROBLEM DEFINITION

본 연구에서 활용한 annotation은 다음과 같다.  
${P}$ = a learner set =($e_1$, $e_2$, ..., $e_m$)  
${E}$ = an exercise set = ($r_1$, $r_2$, ..., $r_m$)  
${K}$ = a problem schema set ($k_1$, $k_2$, ..., $k_m$)  
${R_l}$ = {($e_1$, $r_1$), ($e_2$, $r_2$), ..., ($e_m$, $r_m$)} = 학습자의 문제풀이 sequence

본 연구는 Knowledge Tracing에 Hierarchical Exercise Graph (HEG)를 사용함으로써 문제 간의 관계를 기존 연구보다 더욱 충실하게 반영함으로써 성능을 높이는 것이 주된 contribution이다.  
논문에서 소개하는 HEG는 문제 간의 'direct support relations'와 'indirect support relations'를 보여주는 두 가지의 그래프로 나타난다.  
먼저, direct support relations는 아랫 부분의 그래프에 나타나며 각 노드는 exercise를 나타낸다. 그리고, indirect support relations는 윗 부분의 그래프에 나타나며 각 노드는 problem schema를 나타낸다. 이것을 시각적으로 보면 Figure 3와 같다.
![Figure 3](https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/blob/main/figure3.png?raw=true) 
$Figure 3$  
HEG는 ($A$, $F$, $S_e$)로 표현되며, $A$ $\in$ {0,1}$^{E*E}$는 direct support relations graph의 adjacency matrix이며, $F$ $\in$ $R^{E*t}$는 각 노드의 feature matrix로 노드별로 t개의 feature를 표현하고, $S_e^{E*S}$는 direct support relations graph와 indirect support relations graph의 연결관계를 나타낸다.

## 3. HGKT FRAMEWORK
### 3.1 Framework Overview

HGKT의 framework은 figure 4와 같다.  
![Figure 4](https://github.com/ChuSeongYeub/-KAIST-data_science_and_machine_learning/blob/main/figure4.png?raw=true)
$Figure 4$  
System 1은 HGNN(hierarchical graph neural network)를 통해서 문제 간 hierarchical graph 구조를 학습하여 problem schema embedding을 생성하여 System 2에 전달한다. System 2는 전달받은 embedding value와 hierarchical graph 정보를 활용하여 exercise에 대한 학습자의 지식상태를 예측한다.  

### 3.2 Direct Support Graph Construction
Direct support는 문제 간의 개념과 풀이가 연관된 경우를 나타낸다. Hierarchical graph의 direct support graph는 support relation이 높은 경우와 낮은 경우로 나누어 다음과 같은 방법으로 생성되었다. $Sup$는 문제 간 연관도를 나타내고, $R_{ei}$와 $W_{ei}$는 각각 학생이 정답 혹은 오답을 선택한 경우를 나타낸다.
1. $e_1$과 $e_2$의 support relation이 높은 경우의 각 문제의 조건부 정답률  
   $P(R_{e1}$|$R_{e2}$) > $P(R_{e1}$|$R_{e2}$, $W_{e2}$), $P(W_{e2}$|$W_{e1}$) > $P(W_{e2}$|$R_{e1}$, $W_{e1}$), if $Sup$($e_1$$\rightarrow$$e_2$) > 0
2. $e_1$과 $e_2$의 support relation이 낮은 경우의 각 문제의 조건부 정답률  
    $P(R_{e1}$|$R_{e2}$) = $P(R_{e1}$|$R_{e2}$, $W_{e2}$), $P(W_{e2}$|$W_{e1}$) = $P(W_{e2}$|$R_{e1}$, $W_{e1}$), if $Sup$($e_1$$\rightarrow$$e_2$) = 0

위의 방식에 의해 문제 간 support value를 구성하면 다음과 같다.  
$\bullet$ $Count$(($e_i$, $e_j$) = ($r_i$, $r_j$))는 학생이 $e_j$에 $r_i$를 답하기 전에 $e_i$에 $r_j$를 답한 경우의 수를 의미한다. 분모가 지나치게 작아지는 것을 방지하기 위해 laplacian smoothing parameter로서 $\lambda_p$ = 0.01을 분모에 더했다.  

$P(R_{e1}$|$R_{e2}$) = $Count((e_2, e_1) = (1, 1)) + \lambda_p \over \Sigma_{r_1=0}^1 Count((e_2, e_1) = (1, r_1)) + \lambda_p$  
$P(R_{e1}|R_{e2}, W_{e2}$) = $\Sigma_{r_2=0}^1Count((e_2, e_1) = (r_2, 1)) + \lambda_p \over \Sigma_{r_2=0}^1\Sigma_{r_1=0}^1 Count((e_2, e_1) = (r_2, r_1)) + \lambda_p$ 

위 식에 의해 문제 간 support value는 다음과 같다.  
$Sup$($e_1$$\rightarrow$$e_2$) = max(0, ln$P(R_{e1}|R_{e2}) \over P(R_{e1}|R_{e2}, W_{e2})$) + max(0, ln$P(W_{e2}|W_{e1}) \over P(W_{e2}|R_{e1}, W_{e1})$)
  
### 3.3 Problem Schema Representation Learning
Indirect support는 공통된 problem schema를 가지고 있는 exercise 간의 관계를 그래프로 representation하는 것으로, 이 또한 hierarchical graph로 표현된다. 방법은 다음과 같다.  
먼저, problem schema를 추출하기 위해 BERT[1]를 활용하여 문제의 keyword를 임베딩하고 이것을 hierarchical clustering[2]을 통해 representation하였다. Hierarchical clustering은 각 데이터를 계층에 따라 순차적으로 클러스터링 하는 계층적 군집 분석(agglomerative hierarchical clustering)을 활용한 unsupervised cluster analysis method이다. 이것을 활용한 이유는 임계치(threshold) $\gamma$를 활용하여 그래프의 level 수를 정하고, 이것을 통해 서로 다른 수준의 problem schema를 계층화하여 각 schema에 해당하는 exercise를 군집화하기 위해서이다.  
다음으로, 모든 exercise 간의 관계를 나타내는 direct support graph를 indirect support graph의 problem schema와 fusing하기 위해 DiffPool[3]에서 소개된 assignment matrix($S_e$)로 두 그래프의 연결관계를 표현하였다. $S_e$는 row에 direct support graph의 exercise 노드를 두고, column에 indirect support graph의 problem schema 노드를 두어 두 그래프의 연결관계에 대한 정보를 제   공하는 matrix이다.  
끝으로, exercise와 problem schema 정보를 담고 있는 HEG = ($A, F, S_\gamma$)를 HGNN(hierarchical graph neural networks)을 활용하여 convolution layers와 pooling layers를 통해 direct support graph의 exercise 노드 정보를 공통된 problem schema로 합성곱하여 전파하였다. HGNN은 두 개의 GNN을 통해 두 그래프를 모델링한다. 이와 관련된 구체적인 annotation과 수식은 다음과 같다.  
$A_e$ $\in$ {0, 1}$^{E*E}$= direct graph의 adjacency matrix  
$H_e$ $\in$ $R^{E*t}$= direct graph의 exercise embedding matrix로 node별 feature 표현, $H_o$ = $F \in$ $R^{E*t}$  
$A_s$ $\in$ {0, 1}$^{S*S}$= indirect graph의 adjacency matrix  
$H_s$ $\in$ $R^{S*t}$ = indirect graph의 exercise embedding matrix로 node별 feature 표현  

$H_e^{(l+1)}$ = $GNN_{exer}(A_e, H_e^{(l)})$,  
$A_s$ = $S_\gamma^TA_eS_\gamma$,  
$H_S^{(l+1)} = S_\gamma^TH_e^{(l)}$,  
$H_S^{(l+1)} = GNN_{sche}(A_s, H_s^{(l)})$.

### 3.4 Sequence Modeling Process
HEG를 통해 exercise와 problem schema 관련 정보를 추출한 후 RNNs 기반인 LSTM을 통해 학생의 문제 풀이 sequence에 맞게 각 문제의 정답 여부를 예측한다. 정답 여부 예측의 process는 다음과 같다.

#### 3.4.1 Sequence Propagation
학생이 풀이한 문제의 데이터는 '문제의 개념($v_t$)', '문제 풀이 결과($r_t$)', 그리고 3.3에서 설명한 HEG process를 통해 얻어진 'problem schema($s_t$)'가 joint embedding된 $x_t$를 exercise interaction sequences로 LSTM에 입력하고 그 출력값에 활성화 함수를 적용하여 학생이 problem schema를 습득한 정도를 나타내는 $m_{(t+1)}^{cur}$를 출력한다. 구체적인 수식은 아래와 같고, $W_1과 b_1$는 학습되는 parameter이다.  
$h_{t+1}, c_{t+1}$ = $LSTM(x_{t+1}, h_t, c_t; \theta_{t+1})$,  
$m_{(t+1)}^{cur} = ReLU(W_1 \cdot h_{t+1} + b_1)$.  

#### 3.4.1 Attention Mechanism

HGKT는 두 가지 종류의 attention mechanism(sequence attention, schema attention)을 활용한다.  
sequence attention은 이전까지 유사한 문제를 풀이한 결과를 나타내는 정보로 다음과 같이 표현된다.  
$m_{t+1}^{att} = \Sigma_{i=max(t-\gamma_\beta, 0)}^t\beta_im_i^{cur}, \beta_i = cos(s_{t+1}, s_i)$.  
$\gamma_\beta$는 hyperparameter로 0시점부터 $\gamma_\beta$시점까지의 sequence를 차단함으로써, computational cost를 줄이고, 교육심리학에서 주장하는 학습과정에서의 망각효과[4]를 반영하였다.  
schema attention은 현재 예측하고자 하는 문제와 이전까지의 problem schema의 연관성에 대한 정보로 앞서 설명한 indirect support graph process의 최종 출려값인 $M_{sc} \in R^{k*|S|}$와 problem 간의 유사도를 나타내는 $\alpha_t \in R^{|S|}$를 활용하여 다음과 같이 나타난다. 즉, 한 문제에 대한 정답여부의 정보는 유사한 problem schema를 공유하는 다른 문제 풀이 과정에도 영향을 주도록 한다.  
$m_{t+1}^f = \alpha_{t+1}^Tm_{t+1}^{cur}, \alpha_{t+1} = Softmax(s_{t+1}^TM_{sc})$  
정리하자면, 예측하고자 하는 문제와 관련한 학생의 knowledge mastery 정보($m_{t+1}^{cur}$), 학생의 모든 knowledge mastery 정보($m_{t+1}^{att}$), 그리고 예측하고자 하는 문제와 관련한 학생의 problem schema mastery 정보($m_{t+1}^f$)를 활용하여 학생의 문제 풀이 여부를 예측하는데, 이 3가지 정보는 concat되어 최종 예측값을 출력한다. 이를 식으로 표현하면 다음과 같고, $W_2과 b_2$는 학습되는 parameter이다.  
$\widetilde{r_{t+1}} = \sigma(W_2\cdot[m_{t+1}^{att}, m_{t+1}^{cur}, m_{t+1}^{f}]+b_2)$












REFERENCE

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert:
Pre-training of deep bidirectional transformers for language understanding. arXiv
preprint arXiv:1810.04805 (2018).

[2] Stephen C Johnson. 1967. Hierarchical clustering schemes. Psychometrika 32, 3
(1967), 241–254.

[3] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure
Leskovec. 2018. Hierarchical graph representation learning with differentiable
pooling. In Advances in neural information processing systems. 4800–4810.
[4] Hermann Ebbinghaus. 2013. Memory: A contribution to experimental psychology.
Annals of neurosciences 20, 4 (2013), 155.