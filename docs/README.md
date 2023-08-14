
# Cache-based Recurrent Attention Network

>考虑一个更新的步骤

- 对于模型的输入$X \in (batch\_size, num\_steps, embedding\_dim)$,要对每个timestep输出一个$hidden\_state \in (batch\_size, hidden\_size)$，因此，输出的$h \in (batch\_size, num\_steps, hidden\_size)$
- $num\_steps$ 只能按时间步计算，因此计算中将计算分到每个时间步中，期望的输入和输出变为$X \in (batch\_size, embedding\_dim)$,$h \in (batch\_size,  hidden\_size)$
- 设计一个Memory来存储之前已建模的语义信息。Memory中最多存N个键值对。
$$ \mathcal{M} = \{ < K_n, V_n > \}_{n=1}^{N} $$
>> 其中每个$V_i$代表一个区域（长度为L，未填满用padding填上），$V_i \in (b, L, hidden\_size)$。
>>$K \in ( N,b, d_k)$, $V \in (N,L,b, hidden\_size)$。$d_k$是key的维数,暂时取与$embedding\_dim$相等，若不等则之后查询的第一步可以加一个线性层。

>1. 查询：查询当前时间步需要Attention的区域
$$ weights = softmax(X \cdot K^T)    \ (dim:(b, N)) $$
$$ y =topk (weights)  \ (dim:(b, k)) $$
>>$y \in  (b, k)$, 是取出的k个区域对应的权值， $y_i$对应的区域是$V_i (i = 1,2,3,...,k)$
>> $V_i \in (b, L, hidden\_size)$

>2. 分别做Attention，对于每个 $(y_i, V_i)$
$$ Query = X, Keys = V_i, Values = V_i$$
$$ Q = W_Q X, K = V_i  W_K, V = V_i$$
$$Attn_i = \sum_j softmax(\frac{QK^T}{\sqrt{d_k}})_j \cdot V_{ij} \ (dim: (b,d_h))$$
最后加权求和
$$ Attention  = \sum_{i=1}^k y_i \cdot Attn_i \ (dim: (b, d_h))$$

>3. 更新$hidden\_state$

标准方法：
$$ h = W_h \cdot concat(Attention, X) + b_h \ (dim: (b, d_h)) $$

gated方法(与GRU相同)：
$$ r = \sigma(W_r \cdot concat(Attention, X) + b_r) \ (dim: (b, d_h)) $$
$$ z = \sigma(W_z \cdot concat(Attention, X) + b_z) \ (dim: (b, d_h)) $$
$$ n = tanh(r \cdot (W_nX + b_n) + W_i \cdot Atention + b_i) \ (dim: (b, d_h)) $$
$$ h = (1 - z)\cdot n + z \cdot Attention \ (dim: (b, d_h))$$
>4. 更新Memory

  $V \in (N, L, hidden\_size)$，考察V中是否是满的（即是否还有全零的空位）

	1. V 满

将V中的第一个$V_0$和其对应的 $K_0$ 扔掉。其他 $V_i$ 和其对应的 $K_i$ 向前进一格，空出最后一格。化为不满的情况。

	2. V不满

找出第一个不满的空位，将h填入这个空位，重新计算这个空位所在的$V_i$对应的$K_i$
$$ K_i = Mean(V_i)\  (in\ dimension\ N) $$
>>这里应该考虑其他函数，求均值只是简便考虑

>5. 输出
$$ Output =(W_oh + b_o) $$
