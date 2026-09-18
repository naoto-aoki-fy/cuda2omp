#!/usr/bin/env python3
from support import run_named

if __name__ == '__main__':
 run_named([
  'test_frontend_shim_uses_public_runtime_declarations',
  'test_compilation_database_arguments_and_explicit_override',
  'test_compilation_database_missing_and_ambiguous_diagnostics',
  'test_native_frontend_contract', 'test_native_utf8_and_location_fixture',
  'test_validation_rejections', 'test_macro_launch_rejection_preserves_output', 'test_cuda_declaration_in_header_is_rejected',
 ])
