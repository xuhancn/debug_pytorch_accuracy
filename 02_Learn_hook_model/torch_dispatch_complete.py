import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils._python_dispatch import TorchDispatchMode
import time
from collections import defaultdict


# ==================== 1. Debug and Logging Mode ====================
class DebugMode(TorchDispatchMode):
    """Debug Mode: Log all PyTorch operations"""

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.op_count = 0
        self.stack_log = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.op_count += 1
        kwargs = kwargs or {}

        # Get call stack information (only recent layers)
        import traceback

        stack = traceback.extract_stack()
        caller_info = ""
        for frame in stack[-6:-1]:  # Take the most recent 5 frames
            if "torch_dispatch" not in frame.name:
                caller_info = f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                break

        func_name = func.__name__ if hasattr(func, "__name__") else str(func)

        # Record operation information
        op_info = {
            "id": self.op_count,
            "func": func_name,
            "caller": caller_info,
            "args_info": [],
        }

        # Analyze arguments
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                op_info["args_info"].append(
                    f"arg{i}: Tensor(shape={arg.shape}, dtype={arg.dtype}, "
                    f"grad={arg.requires_grad}, device={arg.device})"
                )
            elif isinstance(arg, (int, float, bool)):
                op_info["args_info"].append(f"arg{i}: {arg}")
            elif isinstance(arg, (list, tuple)):
                op_info["args_info"].append(f"arg{i}: {type(arg).__name__}[{len(arg)}]")
            else:
                op_info["args_info"].append(f"arg{i}: {type(arg).__name__}")

        self.stack_log.append(op_info)

        if self.verbose and self.op_count <= 20:  # Limit output count
            print(f"[{self.op_count:03d}] {func_name} (from {caller_info})")
            for arg_info in op_info["args_info"][:3]:  # Only show first 3 arguments
                print(f"     {arg_info}")

        # Execute original operation
        result = func(*args, **kwargs)

        # Record result information
        if isinstance(result, torch.Tensor):
            op_info["result"] = f"Tensor(shape={result.shape}, dtype={result.dtype})"
            if self.verbose and self.op_count <= 20:
                print(f"     → Result: {op_info['result']}")

        if self.verbose and self.op_count <= 20:
            print(f"     {'─'*60}")

        return result

    def print_summary(self):
        """Print operation statistics summary"""
        print("\n" + "=" * 80)
        print("📊 DEBUG MODE SUMMARY")
        print("=" * 80)
        print(f"Total operations: {self.op_count}")
        print(f"Recorded operations: {len(self.stack_log)}")

        # Count most frequent operations
        op_counter = defaultdict(int)
        for op in self.stack_log:
            op_counter[op["func"]] += 1

        if op_counter:
            print("\nOperation frequency statistics:")
            for op, count in sorted(op_counter.items(), key=lambda x: -x[1])[:10]:
                print(f"  {op:30s}: {count:4d} times")


# ==================== 2. Performance Profiling Mode ====================
class ProfileMode(TorchDispatchMode):
    """Performance Profiling Mode"""

    def __init__(self, track_memory=False):
        super().__init__()
        self.timings = defaultdict(list)
        self.memory_usage = [] if track_memory else None
        self.total_time = 0
        self.call_count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)

        # Record memory usage (if enabled)
        if self.memory_usage is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            else:
                mem_before = 0

        # Measure time
        start_time = time.perf_counter_ns()
        result = func(*args, **(kwargs or {}))
        elapsed_ns = time.perf_counter_ns() - start_time

        # Record memory usage change
        if self.memory_usage is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                mem_change = mem_after - mem_before
                if mem_change != 0:
                    self.memory_usage.append((func_name, mem_change))

        # Record time
        self.timings[func_name].append(elapsed_ns / 1e6)  # Convert to milliseconds
        self.total_time += elapsed_ns
        self.call_count += 1

        return result

    def print_report(self):
        """Print performance report"""
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE PROFILE REPORT")
        print("=" * 80)

        if not self.timings:
            print("No operations recorded.")
            return

        print(
            f"{'Operation':<25} {'Calls':<8} {'Total(ms)':<12} {'Avg(ms)':<12} {'Min(ms)':<10} {'Max(ms)':<10} {'%':<6}"
        )
        print("-" * 80)

        # Calculate statistics
        stats = []
        for op_name, times in self.timings.items():
            total = sum(times)
            avg = total / len(times)
            min_time = min(times)
            max_time = max(times)
            percentage = (
                total / (self.total_time / 1e6)
            ) * 100  # Total time converted to ms

            stats.append(
                (op_name, len(times), total, avg, min_time, max_time, percentage)
            )

        # Sort by total time
        stats.sort(key=lambda x: x[2], reverse=True)

        for op_name, calls, total, avg, min_t, max_t, pct in stats:
            print(
                f"{op_name:<25} {calls:<8} {total:<12.3f} {avg:<12.3f} "
                f"{min_t:<10.3f} {max_t:<10.3f} {pct:<6.1f}"
            )

        print("-" * 80)
        print(f"Total time: {self.total_time / 1e9:.4f} seconds")
        print(f"Total operations: {self.call_count}")

        # Memory usage report
        if self.memory_usage:
            print("\nMemory Changes (CUDA):")
            mem_by_op = defaultdict(int)
            for op_name, mem_change in self.memory_usage:
                mem_by_op[op_name] += mem_change

            for op_name, total_mem in sorted(
                mem_by_op.items(), key=lambda x: -abs(x[1])
            )[:5]:
                sign = "+" if total_mem > 0 else ""
                print(f"  {op_name:<25}: {sign}{total_mem / 1024**2:.2f} MB")


# ==================== 3. Gradient Guard Mode ====================
class GradientGuardMode(TorchDispatchMode):
    """Gradient Guard Mode: Detect potential gradient issues"""

    def __init__(self):
        super().__init__()
        self.warnings = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)

        # Check 1: Modifying requires_grad=True tensors in no_grad()
        if not torch.is_grad_enabled():
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    if func_name in [
                        "add_",
                        "mul_",
                        "sub_",
                        "div_",
                        "copy_",
                    ]:  # in-place operations
                        warning = f"⚠️ In-place operation '{func_name}' in no_grad() on requires_grad tensor!"
                        self.warnings.append(warning)
                        print(warning)

        # Check 2: detach usage
        if func_name == "detach":
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.requires_grad:
                    print(f"🔍 Detaching a requires_grad tensor: shape={arg.shape}")

        # Check 3: Operations that might break computational graph
        if func_name in ["new_tensor", "tensor", "as_tensor"]:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and arg.requires_grad and i > 0:
                    print(
                        f"⚠️ Creating new tensor from requires_grad tensor in {func_name}"
                    )

        return func(*args, **(kwargs or {}))

    def print_warnings(self):
        if self.warnings:
            print(f"\nFound {len(self.warnings)} gradient-related warnings!")


# ==================== 4. Custom Numerical Safety Mode ====================
class NumericalSafetyMode(TorchDispatchMode):
    """Numerical Safety Mode: Prevent numerical issues"""

    def __init__(self, clip_grad_norm=None, eps=1e-8):
        super().__init__()
        self.clip_grad_norm = clip_grad_norm
        self.eps = eps

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)

        # Handle division: avoid division by zero
        if func_name in ["div", "true_divide", "divide"] and len(args) >= 2:
            numerator, denominator = args[0], args[1]
            if isinstance(denominator, torch.Tensor):
                if torch.any(denominator.abs() < self.eps):
                    print(
                        f"⚠️ Near-zero division in {func_name}, adding epsilon={self.eps}"
                    )
                    # Prevent division by zero
                    denominator = torch.where(
                        denominator.abs() < self.eps,
                        torch.sign(denominator) * self.eps,
                        denominator,
                    )
                    args = (numerator, denominator) + args[2:]

        # Handle log: prevent log of non-positive numbers
        elif func_name in ["log", "log10", "log2", "log1p"] and args:
            x = args[0]
            if isinstance(x, torch.Tensor):
                if torch.any(x <= 0) and func_name != "log1p":
                    print(
                        f"⚠️ Non-positive input to {func_name}, clipping to {self.eps}"
                    )
                    x = torch.clamp(x, min=self.eps)
                    args = (x,) + args[1:]

        # Handle gradient clipping
        elif func_name == "_foreach_add_" and self.clip_grad_norm:
            # This is a simplified gradient clipping detection
            print(
                f"📏 Gradient clipping would be applied here (max_norm={self.clip_grad_norm})"
            )

        return func(*args, **(kwargs or {}))


# ==================== 5. Demonstration Model and Training Function ====================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5)
        )

    def forward(self, x):
        return self.net(x)


def train_step(model, data, target, optimizer, criterion):
    """Single training step"""
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss


# ==================== 6. Quantization Mode (for custom operation modification) ====================
class QuantizeMode(TorchDispatchMode):
    """Simulate quantization operations"""

    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.scale = 2 ** (bits - 1) - 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_name = func.__name__

        # Only "quantize" specific operations
        if func_name in ["add", "mul", "matmul"]:
            print(f"🔧 Applying {self.bits}-bit quantization to {func_name}")

            # Simulate quantization: scale and round
            def fake_quantize(tensor):
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    # Simple quantization simulation
                    max_val = tensor.abs().max().item()
                    if max_val > 0:
                        scaled = tensor / max_val * self.scale
                        quantized = torch.round(scaled)
                        dequantized = quantized / self.scale * max_val
                        return dequantized
                return tensor

            # "Quantize" parameters
            new_args = tuple(
                fake_quantize(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )

            result = func(*new_args, **(kwargs or {}))

            # Also "quantize" result
            if isinstance(result, torch.Tensor) and result.is_floating_point():
                result = fake_quantize(result)

            return result

        return func(*args, **(kwargs or {}))
