1D tensor parallel in MLP

## principal

* no 1d-TP

  1. $y=x*A*B$
  2. $grad=dL/dy$

* 1d-TP

  * using $y_i$ to represent $y@rank_i$

  1. $x_i=x_0$
  2. $y_i=x_i*A_i*B_i$, where $A=[A_0,A_1,...,A_7],B=[B_0,B_1,...,B_7]^T$
  3. $y=\sum y_i$
  4. $grad=dL/dy$
  5. $grad_i=grad$

## 1D-TP manual

1. full dataset at all rank(line 58-59)
2. broadcast x(indeed, rank1-7 don't need to implement line68/71)
3. allreduce.SUM(line75)
4. backward

## 1D-TP api

* megatron-LM.MLP, not included in this repo