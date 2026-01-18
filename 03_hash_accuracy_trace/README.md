# hash_accuracy_trace tool

```patch
diff --git a/benchmarks/dynamo/common.py b/benchmarks/dynamo/common.py
index 0f5850058c7..f9cac6e13c4 100644
--- a/benchmarks/dynamo/common.py
+++ b/benchmarks/dynamo/common.py
@@ -2231,16 +2231,23 @@ class BenchmarkRunner:
             model, example_inputs = self.maybe_cast(model, example_inputs)
             accuracy_status = "pass"

+            from hash_accuracy_trace import HashDebugTrace
+
             # Get results of native pytorch
             reset_rng_state()
             model_copy = None
             try:
-                with torch.compiler.set_stance("force_eager"):
-                    model_copy = self.deepcopy_and_maybe_parallelize(model)
-                    self.init_optimizer(name, current_device, model_copy.parameters())
-                    correct_result = self.run_n_iterations(
-                        model_copy, clone_inputs(example_inputs), self.model_iter_fn
-                    )
+                with HashDebugTrace(
+                    log_path=f"trace_first.log",
+                ) as debug_1:
+                    with torch.compiler.set_stance("force_eager"):
+                        model_copy = self.deepcopy_and_maybe_parallelize(model)
+                        self.init_optimizer(
+                            name, current_device, model_copy.parameters()
+                        )
+                        correct_result = self.run_n_iterations(
+                            model_copy, clone_inputs(example_inputs), self.model_iter_fn
+                        )
             except Exception as e:
                 accuracy_status = (
                     "eager_1st_run_OOM"
@@ -2257,12 +2264,17 @@ class BenchmarkRunner:
             reset_rng_state()
             model_copy = None
             try:
-                with torch.compiler.set_stance("force_eager"):
-                    model_copy = self.deepcopy_and_maybe_parallelize(model)
-                    self.init_optimizer(name, current_device, model_copy.parameters())
-                    correct_rerun_result = self.run_n_iterations(
-                        model_copy, clone_inputs(example_inputs), self.model_iter_fn
-                    )
+                with HashDebugTrace(
+                    log_path=f"trace_second.log",
+                ) as debug_2:
+                    with torch.compiler.set_stance("force_eager"):
+                        model_copy = self.deepcopy_and_maybe_parallelize(model)
+                        self.init_optimizer(
+                            name, current_device, model_copy.parameters()
+                        )
+                        correct_rerun_result = self.run_n_iterations(
+                            model_copy, clone_inputs(example_inputs), self.model_iter_fn
+                        )
             except Exception as e:
                 accuracy_status = (
                     "eager_2nd_run_OOM"
```