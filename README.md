# rwkv_padding rwkv的并行推理，通过padding将所有问题等长化。 需要注意的是由于jit问题导致目前有一定的精度损失，目前没有解决

```
lens 记录了每个问题的长度，为了实现state的截断
```
