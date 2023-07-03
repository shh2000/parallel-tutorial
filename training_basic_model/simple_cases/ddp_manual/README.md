Mannully devide input\_data into nproc parts, compare result with ../ddp\_api/

using torchrun to execute

## DDP manually

* what DDP do
  1. broadcast param.data before training
  2. average param.data.grad after backward, before upd param.data
  3. others not related with matmul case
* how to implement data parallen manually
  1. line 67-68 broadcast initial params
  2. line 87-89 all-reduce.sum then /= worldsize

## principal

* no data parallel
  1. $y_0=w_0*x$
  2. $grad=y_0-y$
  3. $w_1=w_0-lr*grad$
* with data parallel
  1. $w_0@rank_i=w_0@rank_0$
  2. $y_0@rank_i=w_0@rank_i*x@rank_i$
  3. $grad@rank_i=y_0@rank_i-y@rank_i$
  4. $grad=\sum grad@rank_i$
  5. $grad@rank_i=grad/gpus$
  6. $w_1@rank_i=w_0@rank_i-lr*grad@rank_i$

Wherein, I replaced the forward propagation process with a simple matrix multiplication and the backward propagation for gradient computation with a simple subtraction.