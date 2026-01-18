import torch
import torch.nn as nn
import torch.optim as optim
from torch_dispatch_complete import (
    DebugMode,
    ProfileMode,
    GradientGuardMode,
    NumericalSafetyMode,
    QuantizeMode,
    SimpleModel,
    train_step,
)


# ==================== 1. Debug Mode Demo ====================
def demo_debug_mode():
    print("=" * 80)
    print("DEMO 1: Debug Mode - Tracking All Operations")
    print("=" * 80)

    x = torch.randn(2, 3, requires_grad=True)
    y = torch.ones(2, 3) * 2

    with DebugMode(verbose=True) as debug:
        # A series of operations
        z = x + y
        z = torch.relu(z)
        z = z.mean()

        # Trigger backward propagation
        z.backward()

        # More operations
        a = torch.randn(3, 4)
        b = torch.randn(4, 2)
        c = a @ b
        d = torch.sigmoid(c)
        e = torch.cat([a, a], dim=1)

    debug.print_summary()


# ==================== 2. Profile Mode Demo ====================
def demo_profile_mode():
    print("\n" + "=" * 80)
    print("DEMO 2: Profile Mode - Performance Analysis")
    print("=" * 80)

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    with ProfileMode(track_memory=False) as profiler:
        # Simulate training loop
        for batch in range(5):
            data = torch.randn(16, 10)  # batch_size=16
            target = torch.randn(16, 5)

            # Training step
            loss = train_step(model, data, target, optimizer, criterion)

            # Some additional operations
            with torch.no_grad():
                # Validation step
                val_data = torch.randn(8, 10)
                val_output = model(val_data)

    profiler.print_report()


# ==================== 3. Gradient Guard Demo ====================
def demo_gradient_guard():
    print("\n" + "=" * 80)
    print("DEMO 3: Gradient Guard Mode")
    print("=" * 80)

    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)

    with GradientGuardMode() as guard:
        print("1. Normal gradient operations:")
        z = x * y
        z = z.sum()
        z.backward()

        print("\n2. Operations in no_grad context:")
        with torch.no_grad():
            # These will trigger warnings
            x.add_(0.1)  # in-place operation
            w = x.detach()
            new_tensor = torch.tensor(x)  # Create new tensor from requires_grad tensor

    guard.print_warnings()


# ==================== 4. Numerical Safety Demo ====================
def demo_numerical_safety():
    print("\n" + "=" * 80)
    print("DEMO 4: Numerical Safety Mode")
    print("=" * 80)

    with NumericalSafetyMode(eps=1e-6) as safety:
        # Risky operations
        a = torch.tensor([1.0, 2.0, 0.0, -1.0])
        b = torch.tensor([2.0, 1.0, 0.0, 1.0])

        print("1. Division with near-zero values:")
        result = a / b  # Will trigger warning
        print(f"Result: {result}")

        print("\n2. Logarithm with non-positive values:")
        values = torch.tensor([1.0, 0.5, 0.0, -0.5])
        log_result = torch.log(values)  # Will trigger warning
        print(f"Result: {log_result}")

        print("\n3. Safe operations:")
        safe_values = torch.tensor([1.0, 2.0, 3.0])
        safe_log = torch.log(safe_values)  # Won't trigger warning
        print(f"Result: {safe_log}")


# ==================== 5. Nested Modes Demo ====================
def demo_nested_modes():
    print("\n" + "=" * 80)
    print("DEMO 5: Nested Modes")
    print("=" * 80)

    print("Nesting multiple modes together:")

    # The order of nesting matters!
    with ProfileMode() as profiler:
        with DebugMode(verbose=False) as debug:
            with NumericalSafetyMode() as safety:
                # Execute some operations
                x = torch.randn(10, 5)
                y = torch.randn(10, 5) * 0.001  # Small values

                z = x / y  # Trigger numerical safety mode
                z = torch.log(z.abs() + 1e-7)  # Trigger numerical safety mode
                result = z.mean()

    print(f"\nProfile results:")
    print(f"Total operations profiled: {profiler.call_count}")
    print(f"Total time: {profiler.total_time / 1e9:.6f} seconds")

    print(f"\nDebug results:")
    print(f"Operations tracked: {debug.op_count}")


# ==================== 6. Custom Operation Modification Demo ====================
def demo_custom_operation_modification():
    print("\n" + "=" * 80)
    print("DEMO 6: Custom Operation Modification")
    print("=" * 80)

    # Use quantization mode
    with QuantizeMode(bits=4) as quantizer:
        a = torch.randn(3, 4) * 10
        b = torch.randn(3, 4) * 10

        print("Original tensors:")
        print(f"a: mean={a.mean():.3f}, std={a.std():.3f}")
        print(f"b: mean={b.mean():.3f}, std={b.std():.3f}")

        print("\nOperations with quantization:")
        c = a + b  # Will trigger quantization
        d = a * b  # Will trigger quantization

        print(f"\nResults:")
        print(f"c: mean={c.mean():.3f}, std={c.std():.3f}")
        print(f"d: mean={d.mean():.3f}, std={d.std():.3f}")


# ==================== 7. Main Function ====================
def main():
    print("🔥🔥🔥 TORCH DISPATCH MODE COMPLETE DEMO 🔥🔥🔥")
    print("PyTorch Version:", torch.__version__)
    print("=" * 80 + "\n")

    # Run all demonstrations
    demo_debug_mode()
    demo_profile_mode()
    demo_gradient_guard()
    demo_numerical_safety()
    demo_nested_modes()
    demo_custom_operation_modification()

    print("\n" + "=" * 80)
    print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # Summary
    print("\n📚 Summary of TorchDispatchMode capabilities:")
    print("1. Debugging: Track all operations")
    print("2. Profiling: Measure performance")
    print("3. Gradient checking: Detect gradient issues")
    print("4. Numerical safety: Prevent numerical problems")
    print("5. Custom modifications: Change operation behavior")
    print("6. Nesting: Combine multiple modes")


# ==================== 8. Individual Demo Runner ====================
def run_demo(demo_name):
    """Run a specific demo by name"""
    demos = {
        "debug": demo_debug_mode,
        "profile": demo_profile_mode,
        "gradient": demo_gradient_guard,
        "numerical": demo_numerical_safety,
        "nested": demo_nested_modes,
        "custom": demo_custom_operation_modification,
        "all": main,
    }

    if demo_name in demos:
        if demo_name == "all":
            demos[demo_name]()
        else:
            demos[demo_name]()
    else:
        print(f"Unknown demo: {demo_name}")
        print(
            "Available demos: debug, profile, gradient, numerical, nested, custom, all"
        )


if __name__ == "__main__":
    # If run directly, execute all demos
    main()
