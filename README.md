# parallel-tutorial
Using simple cases in pytorch to understanding parallel in AI training/inference.

Unless otherwise specified, all code is run in a linux+DGX A100-40GB+nvcr.io/nvidia/pytorch:23.04-py3(pytorch 2.0) environment. 

Please refer to the corresponding installation tutorial for the above environment configuration.

Unless otherwise specified, all code is written by shh2000@github, no code copy from other repos.

Some simple cases in train_basic_model has xx_forward.py, contains only forward(no training) for better understanding.

Cases:

<table>
<thead>
  <tr>
    <th>catagory</th>
    <th>task</th>
    <th>case</th>
    <th>parallel type</th>
    <th>api</th>
    <th>manual with readme</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">train</td>
    <td rowspan="5">simple</td>
    <td rowspan="4">matmul</td>
    <td>None</td>
    <td>/</td>
    <td><a href="https://github.com/shh2000/parallel-tutorial/blob/main/training_basic_model/simple_cases/matmul_full.py">see code</a></td>
  </tr>
  <tr>
    <td>data</td>
    <td>torch.DDP()</td>
    <td><a href="https://github.com/shh2000/parallel-tutorial/blob/main/training_basic_model/simple_cases/ddp_manual/matmul_full.py">see code</a></td>
  </tr>
    <tr>
    <td>1D Tensor</td>
    <td>/</td>
    <td><a href="https://github.com/shh2000/parallel-tutorial/blob/main/training_basic_model/simple_cases/tp_1d_manual/matmul_full.py">see code</a></td>
  </tr>
    </tr>
    <tr>
    <td>Pipeline</td>
    <td>torch Pipe()</td>
    <td>/</td>
  </tr>
</tr>
    <tr>
    <td>C=A*B</td>
    <td>2D-Tensor</td>
    <td>/</td>
    <td><a href="https://github.com/shh2000/parallel-tutorial/blob/main/training_basic_model/simple_cases/tp_2d_manual/matmul_full.py">see code</a></td>
  </tr>
</tbody>
</table>



