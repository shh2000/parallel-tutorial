## 2D TP Principal

* Suppose Y=X*A, X = [[X₀₀, X₀₁], [X₁₀, X₁₁]] and the same as to A
* On GPU 0: Y₁ = X₀₀ * A₀₀ + X₀₁ * A₁₀ 
* On GPU 1: Y₂ = X₀₀ * A₀₁ + X₀₁ * A₁₁ 
* On GPU 2: Y₃ = X₁₀ * A₀₀ + X₁₁ * A₁₀ 
* On GPU 4: Y₄ = X₁₀ * A₀₁ + X₁₁ * A₁₁

* when backward, each GPU calculate grad of parts of A, all_reduce, then update A.

## Code

* It need lot engineer work if really implement 2D-tensor without any API
* So I use a single process, use .to("cuda:0~3") to simulate
* Indeed, calculate of all parts should be implemented at the same time