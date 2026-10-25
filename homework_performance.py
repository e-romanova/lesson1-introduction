"""Решение задания 3: сравнение CPU и CUDA.

По умолчанию запускается безопасный быстрый режим. Для размеров из задания запустите:
    python homework_performance.py --full
Полный batched matmul для (64,1024,1024) требует очень много вычислений и памяти.
"""
import argparse
import json
import time
from pathlib import Path
import torch

QUICK_SHAPES = [(4, 128, 128), (8, 64, 64), (16, 32, 32)]
FULL_SHAPES = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_operation(fn, device: torch.device, repeats: int = 5) -> float:
    """Возвращает среднее время одной операции в миллисекундах."""
    # Прогрев помогает избежать измерения первоначальных накладных расходов.
    fn()
    synchronize(device)
    if device.type == "cuda":
        starts = torch.cuda.Event(enable_timing=True)
        ends = torch.cuda.Event(enable_timing=True)
        starts.record()
        for _ in range(repeats):
            fn()
        ends.record()
        synchronize(device)
        return starts.elapsed_time(ends) / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeats


def benchmark_shape(shape: tuple[int, int, int], repeats: int, devices: list[torch.device]) -> list[dict]:
    # CPU/GPU измеряются на данных одного и того же размера; генерация не входит в замер.
    cpu_a = torch.rand(shape, dtype=torch.float32)
    cpu_b = torch.rand(shape, dtype=torch.float32)
    results = []
    for device in devices:
        a = cpu_a.to(device)
        b = cpu_b.to(device)
        operations = {
            "matmul": lambda: torch.matmul(a, b.transpose(-1, -2)),
            "addition": lambda: a + b,
            "elementwise_multiplication": lambda: a * b,
            "transpose": lambda: a.transpose(-1, -2),
            "sum": lambda: a.sum(),
        }
        for name, fn in operations.items():
            # Для больших CPU/GPU matmul ограничиваем число повторов.
            op_repeats = 1 if name == "matmul" and shape[1] >= 512 else repeats
            ms = time_operation(fn, device, repeats=op_repeats)
            results.append({"shape": list(shape), "operation": name, "device": str(device), "time_ms": round(ms, 4)})
        del a, b
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="использовать точные большие размеры из задания")
    parser.add_argument("--repeats", type=int, default=5, help="число повторов каждой операции в быстром режиме")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats должен быть >= 1")

    shapes = FULL_SHAPES if args.full else QUICK_SHAPES
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("CUDA недоступна: измерения будут выполнены только на CPU.")
    print("Режим:", "полный" if args.full else "быстрый")
    print("Устройства:", ", ".join(map(str, devices)))
    all_results = []
    for shape in shapes:
        print(f"Измерение для shape={shape}...")
        all_results.extend(benchmark_shape(shape, args.repeats, devices))

    # Рассчитываем ускорение только если есть результаты и CPU, и CUDA для одной операции/формы.
    lookup = {(tuple(r["shape"]), r["operation"], r["device"]): r["time_ms"] for r in all_results}
    for r in all_results:
        key = (tuple(r["shape"]), r["operation"])
        cpu_ms = lookup.get((*key, "cpu"))
        gpu_ms = lookup.get((*key, "cuda"))
        r["speedup_cpu_over_gpu"] = round(cpu_ms / gpu_ms, 3) if cpu_ms and gpu_ms else None

    print("\nОперация | Размер | Устройство | мс | Ускорение CPU/GPU")
    for r in all_results:
        speedup = f"{r['speedup_cpu_over_gpu']}x" if r["speedup_cpu_over_gpu"] is not None else "n/a"
        print(f"{r['operation']} | {tuple(r['shape'])} | {r['device']} | {r['time_ms']:.4f} | {speedup}")

    output = Path(__file__).with_name("performance_results.json")
    output.write_text(json.dumps({"cuda_available": torch.cuda.is_available(), "mode": "full" if args.full else "quick", "results": all_results}, indent=2), encoding="utf-8")
    print(f"\nРезультаты сохранены: {output}")


if __name__ == "__main__":
    main()
