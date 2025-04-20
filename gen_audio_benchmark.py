import os
import re
import subprocess

import click
import cpuinfo
import pandas as pd
import torch

cpu_info = cpuinfo.get_cpu_info()
gpu_info = torch.cuda.get_device_name(0)
print(cpu_info)
print(gpu_info)


def run_gen_audio(bs: int):
    cmd = ["python", "gen_audio.py", "--bs", str(bs)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    exec_time = 0.0
    lines = result.stdout.splitlines()

    for i in range(len(lines)):
        if "Infer Speed" in lines[i]:
            match = re.search(r"Infer Speed:\s*([0-9.]+)", lines[i])
        if "Exec Time" in lines[i]:
            match = re.search(r"Exec Time:\s*([0-9.]+)", lines[i])
            if match:
                exec_time = float(match.group(1))
    print(f"BS: {bs}, Naive: {captured_values[-1]} it/s, Exec Time: {exec_time}s")

    if len(captured_values) == 1:
        print(f"No match found for BS {bs} in No CUDA Graph mode.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return captured_values, exec_time


def run_gen_audio_cuda_graph(bs: int, implement: str):
    cmd = ["python", "gen_audio.py", "--cuda-graph", "--bs", str(bs), "--implement", implement]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    exec_time = 0.0
    lines = result.stdout.splitlines()

    for i in range(len(lines)):
        if "Infer Speed" in lines[i]:
            match = re.search(r"Infer Speed:\s*([0-9.]+)", lines[i])
        if "Exec Time" in lines[i]:
            match = re.search(r"Exec Time:\s*([0-9.]+)", lines[i])
            if match:
                exec_time = float(match.group(1))
    print(f"BS: {bs}, CUDA Grpah: {captured_values[-1]} it/s, Exec Time: {exec_time}s")

    if len(captured_values) == 1:
        print(f"No match found for BS {bs} in CUDA Graph mode.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return captured_values, exec_time


def run_gen_audio_compile(bs: int, implement: str):
    cmd = ["python", "gen_audio.py", "--compile", "--bs", str(bs), "--implement", implement]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    exec_time = 0.0
    compile_times = []
    lines = result.stdout.splitlines()

    for i in range(len(lines)):
        if "Infer Speed" in lines[i]:
            match = re.search(r"Infer Speed:\s*([0-9.]+)", lines[i])
        if "T2S Time" in lines[i]:
            match = re.search(r"T2S Time:\s*([0-9.]+)", lines[i])
            if match:
                compile_times.append(float(match.group(1)))
        if "Exec Time" in lines[i]:
            match = re.search(r"Exec Time:\s*([0-9.]+)", lines[i])
            if match:
                exec_time = float(match.group(1))
    compile_times += [0.0, 0.0]
    compile_time = round(compile_times[0] - compile_times[1], 2)
    print(f"BS: {bs}, Compile: {captured_values[-1]} it/s, Compile Time: {compile_time}s, Exec Time: {exec_time}s")

    if len(captured_values) == 1:
        print(f"No match found for BS {bs} in CUDA Graph mode.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return captured_values, compile_time, exec_time


@click.command()
@click.argument("n", type=int)
@click.option("--cuda-graph", is_flag=True, help="Run with CUDA Graph.")
@click.option("--compile", "compile_", is_flag=True, help="Run with Torch Compile.")
@click.option("--implement", type=str, default="flash_attn", help="T2S Decoder Implement For Benchmark")
def main(n=0, cuda_graph=False, compile_=False, implement="Flash_Attn"):
    filename = f"speed_test {cpu_info['brand_raw']} {gpu_info}.csv"
    df = None

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if not df["Batch Size"].equals(pd.Series(range(1, n + 1))):
            print("Batch size mismatch, clearing existing data...")
            df = None

    if df is None:
        df = pd.DataFrame({"Batch Size": list(range(1, n + 1))})

    if "Naive Max It/s" not in df.columns:
        results_naive = []
        results_naive_et = []
        for bs in range(1, n + 1):
            val, et = run_gen_audio(bs)
            results_naive.append(max(val))
            results_naive_et.append(round(et, 2))
        df["Naive Max It/s"] = results_naive
        df["Naive Exec Time"] = results_naive_et

    if cuda_graph:
        results_cuda_graph = []
        results_cuda_graph_et = []
        for bs in range(1, n + 1):
            val, et = run_gen_audio_cuda_graph(bs, implement)
            results_cuda_graph.append(max(val))
            results_cuda_graph_et.append(round(et, 2))
        df[f"{implement} CUDA Graph Max It/s"] = results_cuda_graph
        df[f"{implement} CUDA Graph Exec Time"] = results_cuda_graph_et

    if compile_:
        results_compile = []
        compile_times = []
        results_compile_et = []
        for bs in range(1, n + 1):
            val, ct, et = run_gen_audio_compile(bs, implement)
            results_compile.append(max(val))
            compile_times.append(ct)
            results_compile_et.append(round(et, 2))
        df[f"{implement} Compile  Max It/s"] = results_compile
        df[f"{implement} Compile Exec Times"] = results_compile_et
        df[f"{implement} Compile Times"] = compile_times

    if f"{implement} CUDA Graph Max It/s" in df.columns:
        df[f"Speedup ({implement} CUDA Graph / Naive)"] = [
            round(i, 2) for i in df[f"{implement} CUDA Graph Max It/s"] / df["Naive Max It/s"]
        ]
    if f"{implement} Compile Max It/s" in df.columns:
        df[f"Speedup ({implement} Compile / Naive)"] = [
            round(i, 2) for i in df[f"{implement} Compile Max It/s"] / df["Naive Max It/s"]
        ]

    df.to_csv(filename, index=False)
    print(f"CPU: {cpu_info['brand_raw']} GPU: {gpu_info}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
