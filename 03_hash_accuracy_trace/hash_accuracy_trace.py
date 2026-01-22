import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils._python_dispatch import TorchDispatchMode
import time
from collections import defaultdict
import hashlib
import os
import numpy as np
import sys
from datetime import datetime
from typing import Any, TextIO, Optional, Union, List, Dict
import traceback as tb


def _hash_numpy_array(arr, hash_type="md5"):
    """Helper function: Calculate hash of numpy array"""
    if hash_type == "md5":
        hash_obj = hashlib.md5()
    elif hash_type == "sha1":
        hash_obj = hashlib.sha1()
    elif hash_type == "sha256":
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash type: {hash_type}")

    hash_obj.update(arr.tobytes())
    return hash_obj.hexdigest()


def tensor_to_hash(
    tensor, hash_type="md5", short_hash=False, normalize_low_precision=True
):
    """
    Convert PyTorch tensor to hash value.

    Parameters:
        tensor: PyTorch tensor (supports CPU, CUDA and XPU)
        hash_type: Hash algorithm, options: 'md5', 'sha1', 'sha256'
        short_hash: If True, return first 8 characters of hash
        normalize_low_precision: If True, convert half-precision tensors (FP16/BF16)
                                to float32 before hashing for cross-platform consistency

    Returns:
        Hexadecimal hash string
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be torch.Tensor type")

    is_fp16 = tensor.dtype == torch.float16
    is_bf16 = tensor.dtype == torch.bfloat16
    is_low_precision = is_fp16 or is_bf16

    # Detect device type and handle properly
    device_type = tensor.device.type

    # Ensure tensor is on CPU (supports all device types)
    if device_type == "cuda":
        # CUDA device
        tensor_cpu = tensor.detach().cpu()
        # Wait for copy to complete
        if tensor_cpu.is_pinned():
            torch.cuda.current_stream().synchronize()
    elif device_type == "xpu":
        # XPU device (Intel GPU)
        try:
            # Method 1: Direct .cpu()
            tensor_cpu = tensor.detach().cpu()
            # Synchronize XPU stream
            if hasattr(torch, "xpu"):
                torch.xpu.synchronize()
        except:
            # Method 2: Convert via numpy
            try:
                if is_low_precision and normalize_low_precision:
                    tensor_fp32 = tensor.detach().float()
                    tensor_np = tensor_fp32.numpy(force=True)
                else:
                    tensor_np = tensor.detach().numpy(force=True)
                full_hash = _hash_numpy_array(tensor_np, hash_type)
                return full_hash[:8] if short_hash else full_hash
            except:
                # Method 3: Clone to CPU
                tensor_cpu = tensor.clone().detach().to("cpu")
    else:
        # CPU device
        tensor_cpu = tensor.detach()

    if is_low_precision and normalize_low_precision:
        tensor_cpu = tensor_cpu.float()

    # Convert to numpy and calculate hash
    tensor_np = tensor_cpu.numpy()
    full_hash = _hash_numpy_array(tensor_np, hash_type)
    return full_hash[:8] if short_hash else full_hash


###############################################################################
def get_file_full_path(
    filename: str, base_dir: Optional[str] = None, verify_exists: bool = False
) -> str:
    if base_dir is None:
        base_dir = os.getcwd()

    full_path = os.path.join(base_dir, filename)

    if verify_exists and not os.path.exists(full_path):
        raise FileNotFoundError(f"File is not exist: {full_path}")

    return full_path


def delete_file(file_path):
    """Delete specified file"""
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    try:
        os.remove(file_path)
        print(f"✓ Deleted: {file_path}")
        return True
    except Exception as e:
        print(f"✗ Deletion failed: {e}")
        return False


class FilePrinter:
    """
    A print-like function that outputs to a file.

    Parameters:
        file_name (str or TextIO): File path or file object to write to.
        mode (str): File mode (default: 'a' for append).
        timestamp (bool): Whether to add timestamp to each line.
        echo_to_stdout (bool): Whether to also print to stdout.
        encoding (str): File encoding (default: 'utf-8').
    """

    def __init__(
        self,
        file_name: Union[str, TextIO],
        mode: str = "a",
        timestamp: bool = False,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        echo_to_stdout: bool = False,
        encoding: str = "utf-8",
    ):
        self.timestamp = timestamp
        self.timestamp_format = timestamp_format
        self.echo_to_stdout = echo_to_stdout
        self.encoding = encoding
        self._is_closed = False

        if isinstance(file_name, str):
            full_path = get_file_full_path(file_name)
            delete_file(full_path)
            self.file = open(full_path, mode=mode, encoding=encoding)
            self.need_close = True
        else:
            full_path = get_file_full_path(file_name)
            delete_file(full_path)
            self.file = full_path
            self.need_close = False

        self._write_lock = False  # Prevent recursion when echo_to_stdout=True

    def __call__(
        self,
        *args: Any,
        sep: str = " ",
        end: str = "\n",
        flush: bool = False,
        timestamp: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Print arguments to file.

        Parameters:
            *args: Values to print.
            sep (str): Separator between arguments.
            end (str): String appended after the last value.
            flush (bool): Whether to flush file buffer immediately.
            timestamp (bool): Override default timestamp setting for this call.
            **kwargs: Additional arguments (not used directly, for compatibility).
        """
        if self._is_closed:
            raise ValueError("FilePrinter is closed and cannot write")

        # Use provided timestamp or default
        use_timestamp = timestamp if timestamp is not None else self.timestamp

        # Prepare the message
        message = sep.join(str(arg) for arg in args) + end

        # Add timestamp if needed
        if use_timestamp:
            timestamp_str = datetime.now().strftime(self.timestamp_format)
            message = f"[{timestamp_str}] {message}"

        # Write to file
        try:
            self.file.write(message)
            if flush:
                self.file.flush()
        except (IOError, OSError) as e:
            print(f"Error writing to file: {e}", file=sys.stderr)

        # Echo to stdout if enabled
        if self.echo_to_stdout and not self._write_lock:
            self._write_lock = True
            try:
                sys.stdout.write(message)
                if flush:
                    sys.stdout.flush()
            finally:
                self._write_lock = False

    def write(self, text: str, flush: bool = False) -> None:
        """
        Write raw text to file.

        Parameters:
            text (str): Text to write.
            flush (bool): Whether to flush file buffer immediately.
        """
        if self._is_closed:
            raise ValueError("FilePrinter is closed and cannot write")

        self.file.write(text)
        if flush:
            self.file.flush()

        if self.echo_to_stdout and not self._write_lock:
            self._write_lock = True
            try:
                sys.stdout.write(text)
                if flush:
                    sys.stdout.flush()
            finally:
                self._write_lock = False

    def writeline(self, *args: Any, sep: str = " ") -> None:
        """Write a line (same as calling with default end='\\n')."""
        self(*args, sep=sep, end="\n")

    def flush(self) -> None:
        """Flush the file buffer."""
        if not self._is_closed:
            self.file.flush()
        if self.echo_to_stdout:
            sys.stdout.flush()

    def change_file(
        self, file_name: Union[str, TextIO], mode: str = "a", close_current: bool = True
    ) -> None:
        """
        Change the output file.

        Parameters:
            file_name: New file path or file object.
            mode: File mode for new file.
            close_current: Whether to close current file.
        """
        if close_current and self.need_close and not self._is_closed:
            self.file.close()

        if isinstance(file_name, str):
            self.file = open(file_name, mode=mode, encoding=self.encoding)
            self.need_close = True
            self._is_closed = False
        else:
            self.file = file_name
            self.need_close = False
            self._is_closed = False

    def get_file_info(self) -> dict:
        """Get information about the current output file."""
        info = {
            "timestamp": self.timestamp,
            "echo_to_stdout": self.echo_to_stdout,
            "encoding": self.encoding,
            "file_closable": self.need_close,
            "is_closed": self._is_closed,
        }

        if hasattr(self.file, "name"):
            info["file_name"] = self.file.name
            info["file_mode"] = self.file.mode

        return info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close the file."""
        if self.need_close and not self._is_closed:
            self.file.close()
            self._is_closed = True

    def is_closed(self) -> bool:
        """Check if the file is closed."""
        return self._is_closed

    def reopen(self, mode: str = "a") -> None:
        """Reopen the file if it was closed."""
        if not self._is_closed:
            return

        if self.need_close and hasattr(self.file, "name"):
            self.file = open(self.file.name, mode=mode, encoding=self.encoding)
            self._is_closed = False


# Convenience function similar to print()
def fprint(*args: Any, file_name: Union[str, TextIO] = "output.log", **kwargs) -> None:
    """
    Quick print to file function.

    Parameters:
        *args: Values to print.
        file_name: File path or file object.
        **kwargs: Additional arguments passed to FilePrinter.
    """
    printer = FilePrinter(file_name, **kwargs)
    printer(*args)
    if printer.need_close:
        printer.close()


###############################################################################


class HashDebugTrace(TorchDispatchMode):
    """Debug mode: Log all PyTorch operations with tensor hashes"""

    def __init__(
        self,
        verbose: bool = True,
        log_path: str = "tensor_hash_trace.log",
        max_ops_logged: int = 20000,
        track_gradients: bool = True,
        include_timestamp: bool = False,
        auto_flush: bool = True,
        show_exec_time: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.op_count = 0
        self.stack_log = []
        self.log_path = log_path
        self.max_ops_logged = max_ops_logged
        self.track_gradients = track_gradients
        self.include_timestamp = include_timestamp
        self.auto_flush = auto_flush
        self.show_exec_time = show_exec_time

        # Initialize file printer for logging
        self.printer = FilePrinter(
            log_path, echo_to_stdout=verbose, timestamp=include_timestamp
        )

        # Statistics
        self.stats = {
            "total_ops": 0,
            "tensor_ops": 0,
            "unique_tensors": set(),
            "device_usage": defaultdict(int),
            "operation_types": defaultdict(int),
        }

        # State tracking
        self._is_active = False
        self._file_closed = False

    def _log_message(self, message: str, flush: bool = None) -> None:
        """Log message to file"""
        if flush is None:
            flush = self.auto_flush

        if not self._file_closed:
            try:
                self.printer(message, flush=flush)
            except ValueError as e:
                if "closed" in str(e):
                    print(
                        f"Warning: Log file was closed, cannot write: {message[:50]}..."
                    )
                    self._file_closed = True
                else:
                    raise

    def _get_call_stack(self) -> str:
        """Get formatted call stack information"""
        stack = tb.extract_stack()
        relevant_frames = []

        # Find frames outside torch_dispatch
        for frame in stack[:-1]:  # Exclude current frame
            filename = frame.filename.split("/")[-1]
            if (
                "torch_dispatch" not in frame.name
                and "site-packages" not in frame.filename
                and "lib/python" not in frame.filename
            ):
                relevant_frames.append(f"{filename}:{frame.lineno}")
                if len(relevant_frames) >= 2:  # Get 2 most relevant frames
                    break

        return " <- ".join(relevant_frames) if relevant_frames else "unknown"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.op_count += 1
        kwargs = kwargs or {}

        # Update statistics
        self.stats["total_ops"] += 1
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        self.stats["operation_types"][func_name] += 1

        # Get call stack
        caller_info = self._get_call_stack()

        # Record operation information
        op_info = {
            "id": self.op_count,
            "func": func_name,
            "caller": caller_info,
            "timestamp": time.time(),
            "args_info": [],
            "arg_tensors": [],
            "kwargs_info": dict(kwargs) if kwargs else {},
        }

        # Analyze arguments
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                str_hash = tensor_to_hash(arg, short_hash=False)
                full_hash = tensor_to_hash(arg)
                self.stats["unique_tensors"].add(full_hash)
                self.stats["device_usage"][str(arg.device)] += 1
                self.stats["tensor_ops"] += 1

                # Check for gradients
                grad_info = ""
                if self.track_gradients and arg.grad_fn is not None:
                    grad_info = f", grad_fn={arg.grad_fn.__class__.__name__}"

                arg_info = (
                    f"arg{i}: Tensor(shape={arg.shape}, dtype={arg.dtype}, "
                    f"hash={str_hash}, grad={arg.requires_grad}, "
                    f"device={arg.device}{grad_info})"
                )
                op_info["args_info"].append(arg_info)
                op_info["arg_tensors"].append(
                    {
                        "shape": tuple(arg.shape),
                        "dtype": str(arg.dtype),
                        "hash": str_hash,
                        "requires_grad": arg.requires_grad,
                        "device": str(arg.device),
                    }
                )
            elif isinstance(arg, (int, float, bool, str)):
                op_info["args_info"].append(f"arg{i}: {arg}")
            elif isinstance(arg, (list, tuple)):
                op_info["args_info"].append(f"arg{i}: {type(arg).__name__}[{len(arg)}]")
            elif arg is None:
                op_info["args_info"].append(f"arg{i}: None")
            else:
                op_info["args_info"].append(f"arg{i}: {type(arg).__name__}")

        self.stack_log.append(op_info)

        # Log operation if verbose and within limit
        if self.verbose and self.op_count <= self.max_ops_logged:
            self._log_message(f"[{self.op_count:03d}] {func_name}")
            self._log_message(f"     Called from: {caller_info}")
            for arg_info in op_info["args_info"][:3]:  # Only show first 3 arguments
                self._log_message(f"     {arg_info}")

        # Execute original operation
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            self._log_message(f"     ERROR in {func_name}: {e}")
            raise
        execution_time = time.time() - start_time

        # Record result information
        if isinstance(result, torch.Tensor):
            str_hash = tensor_to_hash(result, short_hash=False)
            full_hash = tensor_to_hash(result)
            self.stats["unique_tensors"].add(full_hash)

            op_info["result"] = {
                "shape": tuple(result.shape),
                "dtype": str(result.dtype),
                "hash": str_hash,
                "requires_grad": result.requires_grad,
                "device": str(result.device),
            }

            if self.verbose and self.op_count <= self.max_ops_logged:
                self._log_message(
                    f"     → Result: Tensor(shape={result.shape}, "
                    f"dtype={result.dtype}, hash={str_hash})"
                )

            if self.show_exec_time:
                self._log_message(f"     Execution time: {execution_time:.6f}s")

        if self.verbose and self.op_count <= self.max_ops_logged:
            self._log_message(f"     {'─'*60}")

        return result

    def print_summary(
        self, to_file: bool = True, file_name: Optional[str] = None
    ) -> None:
        """Print operation statistics summary"""
        summary_lines = [
            "=" * 80,
            "📊 TENSOR HASH DEBUG TRACE SUMMARY",
            "=" * 80,
            f"Total operations: {self.stats['total_ops']}",
            f"Tensor operations: {self.stats['tensor_ops']}",
            f"Unique tensors processed: {len(self.stats['unique_tensors'])}",
            f"Logged operations: {len(self.stack_log)}",
        ]

        # Device usage
        if self.stats["device_usage"]:
            summary_lines.append("\nDevice usage:")
            for device, count in self.stats["device_usage"].items():
                summary_lines.append(f"  {device}: {count} ops")

        # Operation frequency
        if self.stats["operation_types"]:
            summary_lines.append("\nMost frequent operations:")
            for op, count in sorted(
                self.stats["operation_types"].items(), key=lambda x: -x[1]
            )[:10]:
                summary_lines.append(f"  {op:30s}: {count:4d} times")

        # Print summary
        for line in summary_lines:
            if to_file:
                if file_name:
                    # Write to new file
                    with open(file_name, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                elif not self._file_closed:
                    # Write to original log file if it's still open
                    self._log_message(line)
                else:
                    # File is closed, print to stdout instead
                    print(line)
            else:
                print(line)

    def get_tensor_flow_graph(self) -> List[Dict]:
        """Extract tensor flow graph from logged operations"""
        tensor_flow = []
        for op in self.stack_log:
            if "result" in op and op["arg_tensors"]:
                flow_entry = {
                    "operation": op["func"],
                    "op_id": op["id"],
                    "input_tensors": [t["hash"] for t in op["arg_tensors"]],
                    "output_tensor": op["result"]["hash"],
                    "caller": op["caller"],
                }
                tensor_flow.append(flow_entry)
        return tensor_flow

    def find_tensor_by_hash(self, hash_value: str) -> List[Dict]:
        """Find all operations involving a specific tensor hash"""
        matches = []
        for op in self.stack_log:
            # Check in input tensors
            for tensor_info in op.get("arg_tensors", []):
                if tensor_info["hash"] == hash_value:
                    matches.append(
                        {
                            "operation": op["func"],
                            "op_id": op["id"],
                            "role": "input",
                            "tensor_info": tensor_info,
                            "caller": op["caller"],
                        }
                    )

            # Check in result
            if "result" in op and op["result"]["hash"] == hash_value:
                matches.append(
                    {
                        "operation": op["func"],
                        "op_id": op["id"],
                        "role": "output",
                        "tensor_info": op["result"],
                        "caller": op["caller"],
                    }
                )

        return matches

    def export_to_json(self, file_name: str = "trace_export.json") -> None:
        """Export trace data to JSON file"""
        import json

        export_data = {
            "metadata": {
                "total_operations": self.stats["total_ops"],
                "tensor_operations": self.stats["tensor_ops"],
                "unique_tensors": len(self.stats["unique_tensors"]),
                "log_file": self.log_path,
                "export_timestamp": datetime.now().isoformat(),
            },
            "statistics": {
                "total_ops": self.stats["total_ops"],
                "tensor_ops": self.stats["tensor_ops"],
                "unique_tensors": len(self.stats["unique_tensors"]),
                "device_usage": dict(self.stats["device_usage"]),
                "operation_types": dict(self.stats["operation_types"]),
            },
            "tensor_flow": self.get_tensor_flow_graph(),
            "operations": self.stack_log,
        }

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        # Log to stdout instead of trying to write to closed file
        print(f"Trace data exported to {file_name}")

    def __enter__(self):
        self._is_active = True
        self._file_closed = False
        self._log_message("\n" + "=" * 80)
        self._log_message("🚀 STARTING TENSOR HASH DEBUG TRACE")
        self._log_message("=" * 80)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Print summary (will handle closed file case)
        self.print_summary(to_file=True)

        # Close the file printer
        if hasattr(self, "printer"):
            self.printer.close()
            self._file_closed = True

        self._is_active = False

        # Call parent's exit
        return super().__exit__(exc_type, exc_val, exc_tb)

    def is_active(self) -> bool:
        """Check if the trace is currently active."""
        return self._is_active
