import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from utils import get_benchmark_dataloaders, get_benchmark_model

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

    # model = torch.compile(model)
    model.eval()
    model = model.to(device)
    model: WorldMachine

    loader, _ = get_benchmark_dataloaders()

    item = next(iter(loader))
    item = item.to(device)

    state = torch.zeros([32, 100, 128], device=device)

    with torch.no_grad():
        if torch_profiler:
            prof = profile(activities=[
                           ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True)
            prof.__enter__()

        for _ in range(2):
            model.inference(
                state, sensory_data=item["inputs"], sensory_masks=item["input_masks"])

            torch.cuda.synchronize()

        if torch_profiler:
            prof.__exit__(None, None, None)

    if torch_profiler:
        prof.export_chrome_trace("bench_inference_trace.json")

        print(prof.key_averages().table())
