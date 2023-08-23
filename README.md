# rwkv_padding rwkv的并行推理，通过padding将所有问题等长化。 需要注意的是由于jit问题导致目前有一定的state小数点后几位有细微差别，目前没有解决，不开jit时state完全一致
padding操作可参考hf 官网教程

gen.py 提供了使用实例，参考gen_bsz 根据自己的需求修改generate函数
```
    answer, state = pipeline.gen_bsz(msg, token_count=500, args=args)

```
lengs 记录了每个问题的长度，为了实现state的截断。所以记得传递lengs参数，当seq推理结束后lengs可随意设置
```
tokens, lengs = self.encode_bsz(ctx) if i == 0 else (token, torch.tensor(1))
while len(tokens[0]) > 0:
    out, state = self.model.forward(tokens[:, :args.chunk_len], state, lengs)
    tokens = tokens[:, args.chunk_len:]

```
## 最后再提醒一句，padding实现的并行推理会有一定程度的浪费，每个batch中的数据最好接近于等长减少浪费。后续可以等待rwkv群里大佬修改kernel去除padding浪费问题。
