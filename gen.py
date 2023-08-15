import os, sys, torch, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
CHAT_LANG = 'Chinese'
from rwkv_bsz.model_bsz import RWKV # pip install rwkv
from rwkv_bsz.utils import PIPELINE, PIPELINE_ARGS
model = RWKV(model='D:/ChatRWKV-main/model/RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096', strategy='cuda fp16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                     alpha_frequency = 0.0,
                     alpha_presence = 0.0,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################
msg1 = ["Q: 你是谁？\n\nA:",
        "Q: 你好\n\nA:",
        "Q: 西瓜是什么\n\nA:",
        "Q: 你有多厉害？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: hi\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 将patient翻译为中文\n\nA:",
        "Q: What\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 运动后不出汗，皮肤发热　运动后不出汗，皮肤发热经常这样没有\n\nA:",
        "Q: What\n\nA:",
        "Q: 你是谁？\n\nA:",
        "Q: 你好\n\nA:",
        "Q: 你好\n\nA:",
        "Q: hi\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: hi\n\nA:",
        "Q: 你能做什么？\n\nA:",
        "Q: 你好\n\nA:",
        ]
msg2 = ["Q: 你好\n\nA:"]
msg3 = ["Q: What\n\nA:"]
def gen(msg):
    answer1 = pipeline.gen_bsz(msg, token_count=500, args=args)
    print(answer1)
    #print(answer1[0,:4])
    return answer1

# start = time.time()
# print(pipeline.encode_bsz(msg1))
# end = time.time()
# print(end-start)
#

gen(msg2)

start1 = time.time()
gen(msg1)
end1 = time.time()
print(end1-start1)

# msg1  = ["你好"]
#
# gen(msg1)

# start = time.time()
# gen(msg1)
# end = time.time()
# print(end-start)
#
# start1 = time.time()
# gen(msg2)
# end1 = time.time()
# print(end1-start1)

