import datetime
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
    lines = result.stdout.splitlines()

    for i in range(len(lines) - 1):
        if "T2S Decoding EOS" in lines[i]:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lines[i + 1])
            if match:
                captured_values.append(float(match.group()))

    if len(captured_values) == 1:
        print(f"No match found for BS {bs} in No CUDA Graph mode.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return captured_values


def run_gen_audio_cuda_graph(bs: int, implement: str):
    cmd = ["python", "gen_audio.py", "--cuda-graph", "--bs", str(bs), "--implement", implement]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    lines = result.stdout.splitlines()

    for i in range(len(lines) - 1):
        if "T2S Decoding EOS" in lines[i]:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lines[i + 1])
            if match:
                captured_values.append(float(match.group()))

    if len(captured_values) == 1:
        print(f"No match found for BS {bs} in CUDA Graph mode.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return captured_values


def run_gen_audio_compile(bs: int, implement: str):
    cmd = ["python", "gen_audio.py", "--compile", "--bs", str(bs), "--implement", implement]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    lines = result.stdout.splitlines()
    compile_time = 0.0

    for i in range(len(lines) - 1):
        if "T2S Decoding EOS" in lines[i]:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lines[i + 1])
            if match:
                captured_values.append(float(match.group()))
        if "Compile Time" in lines[i]:
            match = re.search(r"Compile Time:\s*([0-9]+\.[0-9]{2})", lines[i])
            if match:
                compile_time = float(match.group(1))

    if len(captured_values) == 1:
        print(f"No match found for BS {bs} in CUDA Graph mode.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return captured_values, compile_time


@click.command()
@click.argument("n", type=int)
@click.option("--cuda-graph", is_flag=True, help="Run with CUDA Graph.")
@click.option("--compile", "compile_", is_flag=True, help="Run with Torch Compile.")
@click.option("--implement", type=str, default="Flash_Attn", help="T2S Decoder Implement For Benchmark")
def main(n=0, cuda_graph=False, compile_=False, implement="Flash_Attn"):
    filename = f"speed_test_{cpu_info['brand_raw']}_{gpu_info}_implement.csv"
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
        for bs in range(1, n + 1):
            val = max(run_gen_audio(bs))
            results_naive.append(val)
        df["Naive Max It/s"] = results_naive

    if cuda_graph:
        results_cuda_graph = []
        for bs in range(1, n + 1):
            val = max(run_gen_audio_cuda_graph(bs, implement))
            results_cuda_graph.append(val)
        df["CUDA Graph Max It/s"] = results_cuda_graph

    if compile_:
        results_compile = []
        compile_times = []
        for bs in range(1, n + 1):
            val, ct = run_gen_audio_compile(bs, implement)
            results_compile.append(max(val))
            compile_times.append(ct)
        df["Compile Max It/s"] = results_compile
        df["Compile Times"] = compile_times

    # 更新加速比
    if "CUDA Graph Max It/s" in df.columns:
        df["Speedup (CUDA Graph / Naive)"] = df["CUDA Graph Max It/s"] / df["Naive Max It/s"]
    if "Compile Max It/s" in df.columns:
        df["Speedup (Compile / Naive)"] = df["Compile Max It/s"] / df["Naive Max It/s"]

    df.to_csv(filename, index=False)
    print(f"CPU: {cpu_info['brand_raw']} GPU: {gpu_info}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
