import torch
import torch.nn as nn
import torch.optim as optim
from hash_accuracy_trace import HashDebugTrace


def setup_deterministic():
    """Setup deterministic execution"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # For specific operations, you can set warnings:
    torch.use_deterministic_algorithms(True, warn_only=True)


def demo_debug_mode():
    """Demonstrate the HashDebugTrace functionality"""
    print("Testing HashDebugTrace...")
    setup_deterministic()

    # Create tensors on different devices if available
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    if torch.xpu.is_available():
        devices.append("xpu")

    for device_name in devices:
        print(f"\nRunning on device: {device_name}")
        device = torch.device(device_name)

        with HashDebugTrace(
            verbose=True, log_path=f"trace_{device_name}.log", max_ops_logged=50
        ) as debug:
            # A series of operations
            x = torch.randn(2, 3, requires_grad=True).to(device)
            y = torch.ones(2, 3).to(device) * 2

            z = x + y
            z = torch.relu(z)
            z = z.mean()

            # Trigger backpropagation
            if x.requires_grad:
                z.backward()

            # More operations
            a = torch.randn(3, 4).to(device)
            b = torch.randn(4, 2).to(device)
            c = a @ b
            d = torch.sigmoid(c)
            e = torch.cat([a, a], dim=1)

            # Matrix operations (only on CPU for inverse)
            if device_name == "cpu":
                mat = torch.randn(5, 5).to(device)
                try:
                    inv = torch.inverse(mat)
                except Exception as e:
                    print(f"Matrix inverse failed: {e}")

            # View and reshape operations
            f = a.view(-1)
            g = b.reshape(2, 4)

        # Export to JSON - this will now work because we print instead of trying to write to closed file
        debug.export_to_json(f"trace_export_{device_name}.json")

        # Find specific tensor (example)
        if debug.stack_log and debug.stack_log[0].get("arg_tensors"):
            first_tensor_hash = debug.stack_log[0]["arg_tensors"][0]["hash"]
            matches = debug.find_tensor_by_hash(first_tensor_hash)
            print(f"Tensor {first_tensor_hash} appears in {len(matches)} operations")


def analyze_tensor_flow():
    """Analyze tensor flow in a simple neural network"""
    print("\n" + "=" * 80)
    print("ANALYZING TENSOR FLOW IN NEURAL NETWORK")
    print("=" * 80)

    with HashDebugTrace(
        verbose=True,
        log_path="nn_trace.log",
        max_ops_logged=200,
        include_timestamp=False,
    ) as trace:
        # Simple neural network
        model = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5), nn.Softmax(dim=1)
        )

        # Create input
        batch_size = 4
        input_data = torch.randn(batch_size, 10)

        # Forward pass
        output = model(input_data)

        # Loss calculation
        target = torch.randint(0, 5, (batch_size,))
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

    # Print summary
    trace.print_summary(to_file=False)

    # Export detailed analysis
    trace.export_to_json("nn_trace_export.json")

    # Get tensor flow graph
    tensor_flow = trace.get_tensor_flow_graph()
    print(f"\nGenerated {len(tensor_flow)} tensor flow entries")

    # Find all tensors created by linear layers
    linear_ops = [op for op in tensor_flow if "linear" in op["operation"].lower()]
    print(f"Linear operations: {len(linear_ops)}")


if __name__ == "__main__":
    try:
        # Run demos
        demo_debug_mode()
        analyze_tensor_flow()

        print("\n" + "=" * 80)
        print("✅ All demos completed. Check the generated log files.")
        print("=" * 80)
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback

        traceback.print_exc()
