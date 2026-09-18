#!/usr/bin/env python3
from support import run_named

if __name__ == '__main__':
 run_named([
  'test_call_graph_selects_only_barrier_reaching_coroutines',
  'test_canonical_declarations_disambiguate_names_and_overloads',
  'test_generated_identifiers_are_stable_and_collision_free',
  'test_in_place_declarations_and_lexical_contexts_compile',
  'test_runtime_include_and_shared_state_stay_in_place',
  'test_lexical_context_and_utf8_before_kernel', 'test_complex_launch_expressions',
 ])
