import datetime
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


def run_gen_audio_no_cuda_graph(bs: int):
    cmd = ["python", "gen_audio.py", "--bs", str(bs)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    lines = result.stdout.splitlines()

    for i in range(len(lines) - 1):
        if "T2S Decoding EOS" in lines[i]:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lines[i + 1])
            if match:
                captured_values.append(float(match.group()))

    return captured_values


def run_gen_audio(bs: int):
    cmd = ["python", "gen_audio.py", "--cuda-graph", "--bs", str(bs)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    captured_values = [0.0]
    lines = result.stdout.splitlines()

    for i in range(len(lines) - 1):
        if "T2S Decoding EOS" in lines[i]:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", lines[i + 1])
            if match:
                captured_values.append(float(match.group()))

    return captured_values


@click.command()
@click.argument("n", type=int)
def main(n=0):
    results_no_cuda = {}
    results_cuda = {}

    for bs in range(1, n + 1):
        captured_values_no_cuda = run_gen_audio_no_cuda_graph(bs)
        captured_values_cuda = run_gen_audio(bs)

        if captured_values_no_cuda:
            results_no_cuda[bs] = max(captured_values_no_cuda)
        else:
            results_no_cuda[bs] = 0

        if captured_values_cuda:
            results_cuda[bs] = max(captured_values_cuda)
        else:
            results_cuda[bs] = 0

        print(f"BS {bs} - No CUDA Graph: {results_no_cuda[bs]} it/s, CUDA Graph: {results_cuda[bs]} it/s")

    df = pd.DataFrame(
        {
            "Batch Size": list(results_no_cuda.keys()),
            "No CUDA Graph Max It/s": list(results_no_cuda.values()),
            "CUDA Graph Max It/s": list(results_cuda.values()),
        }
    )

    df["Speedup (CUDA Graph / No CUDA Graph)"] = df["CUDA Graph Max It/s"] / df["No CUDA Graph Max It/s"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"speed_test_{cpu_info['brand_raw']}_{gpu_info}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print(f"CPU: {cpu_info['brand_raw']} GPU: {gpu_info}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
