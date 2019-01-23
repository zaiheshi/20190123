#### Bilingual Expert <= self-attention mechanism + transformer neural networks

##### Quality Estimation

- traditional solution : formulate the sentence level score or word level tags prediction as a constraint regression or sequence labeling problem respectively
- promising solution : build a **multi-task learning model** to incorporate quality estimation task with **automatic post-editing (APE)** together
- 许多双语余料库　==> 很容易地建立**条件语言模型**作为一个强大的特征提取器。+　4-dimensional token mis-matching features(measuring the difference between what the bilingual expert model will predict and the actual token of machine translation output.)
  - 好的src与mt
  - 差的src与mt 

- 抽象表示：p(t|s) = p(t|z)p(z|s)，代表的大概是该系统可以由encoder与decoder系统组合而成，(s->z)&&(z->t)
  - s==>source sentence    t==>target sentence     目标词指的是什么？
  - z==>**latent variable to represent the encoded source sentence** 
  - 使用了encoder-decoder模型？
- training data : (s, m, t, h, y) 
  - t : post-edited sentence 
  - h : hter score
  - y : "ok/bad" label for sentence
- task : learn a regression model p(h|s, m) and a sequence labeling model p(y|s, m)
  - p(h|s, m) : 根据源句(s)与翻译句子(m)来预测hter得分，回归问题
  - p(y|s, m) : 根据源句(s)与翻译句子(m)来进行序列标注

##### Bilingual Expert Model

- parallel corpus including (s, t) pairs to train a neural bilingual expert model

- 根据公式p(t|s) = p(t|z)p(z|s)遇到的问题是：p(t|z)p(z|s)是未知的，但是需要了解z(包含源句与翻译句子之间的高层语义信息)

  - 解决方法：Bayes rule...后验概率 
    $$
    P(z|st)=\frac{P(t|z)P(z|s)}{P(t|s)}
    $$

- ...........................

##### Bidirectional Transformer

- Transformer is based on attention mechanisms(**dispensing with recurrence and convolution**)

- advantages of self-attention mechanism

  - its gating or multiplication enables crisp error propagation
  - it can replace sequence-aligned recurrence entirely
  - it is trivial to be parallelized during training

- three models()

  1. self-attention encoder for the source sentence

  2. forward and backward self-attention encoders for target sentence

  3. the reconstructor for the target sentence

  - he first two modules represent the proposed posterior approximation P(z|st) and the third reconstruction process corresponds to p(t|z)

- 


