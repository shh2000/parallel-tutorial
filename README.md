# parallel-tutorial
Using simple cases in pytorch to understanding parallel in AI training/inference.

Unless otherwise specified, all code is run in a linux+DGX A100-40GB+nvcr.io/nvidia/pytorch:23.04-py3(pytorch 2.0) environment. 

Please refer to the corresponding installation tutorial for the above environment configuration.

Unless otherwise specified, all code is written by shh2000@github, no code copy from other repos.

Some simple cases in train_basic_model has xx_forward.py, contains only forward(no training) for better understanding.

Cases:

```html
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh">catagory</th>
    <th class="tg-baqh">task</th>
    <th class="tg-baqh">case</th>
    <th class="tg-baqh">parallel type</th>
    <th class="tg-baqh">api</th>
    <th class="tg-baqh">manual with readme</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh" rowspan="2">train</td>
    <td class="tg-baqh" rowspan="2">simple</td>
    <td class="tg-baqh" rowspan="2">matmul</td>
    <td class="tg-baqh">None</td>
    <td class="tg-baqh">/</td>
    <td class="tg-baqh">https://github.com/shh2000/parallel-tutorial/blob/main/training_basic_model/simple_cases/matmul_full.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">data</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">https://github.com/shh2000/parallel-tutorial/blob/main/training_basic_model/simple_cases/ddp_manual/matmul_full.py</td>
  </tr>
</tbody>
</table>
```