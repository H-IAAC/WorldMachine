import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from utils import get_benchmark_dataloaders, get_benchmark_model, get_trainer

from world_machine import WorldMachine
from world_machine.profile import profile_range

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_profiler", "--tp",
                        action="store_true", default=False)
    args = parser.parse_args()

    torch_profiler = args.torch_profiler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with profile_range("create_model", category="benchmark", domain="world_machine"):
        model = get_benchmark_model()
        torch.cuda.synchronize()

    # model = torch.jit.script(model)
    # model = torch.compile(model)
    model.eval()
    model = model.to(device)
    model: WorldMachine
    model.pre_compute_attention_bias(100)

    loader, _ = get_benchmark_dataloaders(2)
    loaders = {"train": loader, "val": loader}

    trainer = get_trainer()
    optim = torch.optim.AdamW(model.parameters())
    torch.cuda.synchronize()

    if torch_profiler:
        prof = profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True)
        prof.__enter__()

    trainer(model, loaders, optim, 3)
    torch.cuda.synchronize()

    if torch_profiler:
        prof.__exit__(None, None, None)

    if torch_profiler:
        prof.export_chrome_trace("bench_train_trace.json")

        print(prof.key_averages().table())
