# import argparse
# from tot.methods.bfs import solve
# from tot.tasks.game24 import Game24Task

# #args = argparse.Namespace(backend=r'/home/czy/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

# args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)


# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0]) 

import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task

#args = argparse.Namespace(backend='local_model', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

args = argparse.Namespace(
    backend='local/llama-2-7b-hf',  # 关键修改：以 "local/" 开头
    temperature=0.7,
    task='game24',
    naive_run=True,
    prompt_sample=None,
    method_generate='propose',
    method_evaluate='value',
    method_select='greedy',
    n_generate_sample=1,
    n_evaluate_sample=3,
    n_select_sample=5
)

task = Game24Task()
ys, infos = solve(args, task, 900)
print(ys[0])