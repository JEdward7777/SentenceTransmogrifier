#!/usr/bin/env python3
"""
Test script to verify the case-change feature works correctly.
This tests that uppercase and lowercase operations are properly detected and applied.
"""

import sys
sys.path.insert(0, 'src')

from sentence_transmorgrifier.transmorgrify import _trace_edits, _list_trace, UPPERCASE, LOWERCASE, MATCH

def test_case_changes():
    """Test various case change scenarios"""
    
    print("Testing case-change detection in edit tracing...\n")
    
    # Test 1: Simple lowercase to uppercase
    print("Test 1: 'hello' -> 'HELLO'")
    trace = _trace_edits("hello", "HELLO", print_debug=True)
    trace_list = _list_trace(trace)
    actions = [t.action for t in trace_list if t.action != 3]  # Exclude START
    print(f"Actions: {actions}")
    print(f"Expected all UPPERCASE (4): {all(a == UPPERCASE for a in actions)}")
    print()
    
    # Test 2: Simple uppercase to lowercase
    print("Test 2: 'WORLD' -> 'world'")
    trace = _trace_edits("WORLD", "world", print_debug=True)
    trace_list = _list_trace(trace)
    actions = [t.action for t in trace_list if t.action != 3]
    print(f"Actions: {actions}")
    print(f"Expected all LOWERCASE (5): {all(a == LOWERCASE for a in actions)}")
    print()
    
    # Test 3: Mixed case changes
    print("Test 3: 'HeLLo' -> 'hEllO'")
    trace = _trace_edits("HeLLo", "hEllO", print_debug=True)
    trace_list = _list_trace(trace)
    for t in trace_list:
        if t.action != 3:  # Skip START
            action_name = {0: "MATCH", 1: "DELETE", 2: "INSERT", 4: "UPPERCASE", 5: "LOWERCASE"}.get(t.action, "UNKNOWN")
            print(f"  {action_name}: '{t.char}'")
    print()
    
    # Test 4: No changes (all match)
    print("Test 4: 'test' -> 'test'")
    trace = _trace_edits("test", "test", print_debug=True)
    trace_list = _list_trace(trace)
    actions = [t.action for t in trace_list if t.action != 3]
    print(f"Actions: {actions}")
    print(f"Expected all MATCH (0): {all(a == MATCH for a in actions)}")
    print()
    
    # Test 5: Mixed operations
    print("Test 5: 'hello world' -> 'HELLO WORLD!'")
    trace = _trace_edits("hello world", "HELLO WORLD!", print_debug=True)
    trace_list = _list_trace(trace)
    for t in trace_list:
        if t.action != 3:
            action_name = {0: "MATCH", 1: "DELETE", 2: "INSERT", 4: "UPPERCASE", 5: "LOWERCASE"}.get(t.action, "UNKNOWN")
            print(f"  {action_name}: '{t.char}'")
    print()
    
    print("=" * 60)
    print("Case-change feature test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_case_changes()