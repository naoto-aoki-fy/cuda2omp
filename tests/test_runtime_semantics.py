#!/usr/bin/env python3
from support import run_named

if __name__ == '__main__':
 run_named([
  'test_cpu_cuda_runtime_api_and_error_state',
  'test_cpu_cuda_runtime_rejects_invalid_and_overflowing_operations',
  'test_all_semantics', 'test_invalid_barrier_reports_runtime_diagnostic', 'test_cuda_launch_kernel',
  'test_forward_declarations_preserve_coroutine_classification_and_returns',
 ])
