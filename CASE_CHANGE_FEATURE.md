# Case-Change Feature Implementation

## Overview
This document describes the implementation of the case-change feature for the sentence transmogrifier project. This feature allows the model to learn and apply uppercase and lowercase transformations as discrete operations, rather than treating them as delete+insert operations.

## Motivation
Previously, changing the case of a letter (e.g., 'a' to 'A') required two operations:
1. DELETE the lowercase letter
2. INSERT the uppercase letter

This approach had limitations:
- The model needed to see all possible letter combinations during training
- Poor generalization to unseen letter pairs
- Inefficient use of model capacity

With the new case-change feature, case transformations are recognized as a single operation, allowing the model to:
- Learn the concept of "uppercase" and "lowercase" as operations
- Generalize to any alphabetic character
- Reduce training data requirements

## Implementation Details

### 1. New Action Constants
Added two new action types to the existing set (MATCH, DELETE_FROM, INSERT_TO, START):
- `UPPERCASE = 4` - Transform character to uppercase
- `LOWERCASE = 5` - Transform character to lowercase

### 2. Case-Change Cost
Introduced a small cost constant for case-change operations:
- `CASE_CHANGE_COST = 0.001`

This cost is:
- Much smaller than insert/delete operations (cost = 1)
- Slightly larger than match operations (cost = 0)
- Ensures case-change is preferred over insert/delete but not over exact match

### 3. File Version Update
Incremented `FILE_VERSION` from 1 to 2 to indicate the new feature and maintain compatibility tracking.

### 4. Edit Tracing Enhancement
Modified `_trace_edits()` function to detect case-only differences:
- Checks if characters differ only in case using: `char1.lower() == char2.lower() and char1 != char2`
- Determines action based on target character: `UPPERCASE if to_char.isupper() else LOWERCASE`
- Applied to all characters (non-alphabetic characters pass through unchanged)
- Uses `<=` comparison to prefer case-change over insert/delete operations

### 5. Training Data Collection
Updated `_parse_single_for_training()` to handle new actions:
- UPPERCASE: Consumes character from input, adds uppercased version to output
- LOWERCASE: Consumes character from input, adds lowercased version to output
- Both reset `continuous_added` and `continuous_dropped` counters (like MATCH)
- Stores space character in 'char' field (since char model isn't used for these actions)

### 6. Inference Implementation
Modified `_do_reconstruct()` to execute case-change operations:
- UPPERCASE: Applies `.upper()` to current character
- LOWERCASE: Applies `.lower()` to current character
- Both consume from input and add to output
- Both update `used_from` tracking
- Both reset continuous operation counters

### 7. Debug Output
Enhanced debugging functions:
- `__str__()`: Shows `<upper> {char}` and `<lower> {char}`
- `_diffs_to_str()`: Uses `^-{char}` for UPPERCASE and `v-{char}` for LOWERCASE

## Testing

### Test 1: Basic Case Detection
Verified that the edit tracing correctly identifies case-change operations:
- 'hello' → 'HELLO': All UPPERCASE operations ✓
- 'WORLD' → 'world': All LOWERCASE operations ✓
- 'test' → 'test': All MATCH operations ✓

### Test 2: Full Integration - Uppercase Transformation
Trained model to convert lowercase to uppercase:
- Training: 5 sentences (lowercase → UPPERCASE)
- Inference: Successfully converted new inputs to uppercase
- All test cases matched expected output ✓

### Test 3: Full Integration - Lowercase Transformation
Trained model to convert uppercase to lowercase:
- Training: 3 sentences (UPPERCASE → lowercase)
- Inference: Successfully converted new inputs to lowercase
- All test cases matched expected output ✓

### Test 4: Full Integration - Title Case
Trained model to capitalize first letter of each word:
- Training: 5 sentences (lowercase → Title Case)
- Inference: Successfully applied title case to new inputs
- All test cases matched expected output ✓

## Benefits

1. **Improved Generalization**: Model learns the concept of case transformation rather than memorizing character pairs
2. **Reduced Training Data**: No need to see all possible letter combinations
3. **Better Efficiency**: Single operation instead of delete+insert
4. **Clearer Semantics**: Case changes are explicitly represented in the edit stream
5. **Backward Compatible**: File version tracking ensures old models still work

## Files Modified

- `src/sentence_transmorgrifier/transmorgrify.py`: Core implementation
- `test_case_change.py`: Basic edit tracing tests
- `test_case_change_full.py`: Full integration tests with training and inference

## Usage Example

```python
from sentence_transmorgrifier.transmorgrify import Transmorgrifier

# Create and train model for uppercase transformation
tm = Transmorgrifier()
tm.train(
    from_sentences=['hello world', 'test case'],
    to_sentences=['HELLO WORLD', 'TEST CASE'],
    iterations=100
)

# Apply to new data
results = list(tm.execute(['new input']))
# Output: ['NEW INPUT']
```

## Technical Notes

- Case-change operations work on all characters, but only affect alphabetic ones
- Non-alphabetic characters pass through unchanged when `.upper()` or `.lower()` is called
- The small cost (0.001) ensures case-change is preferred over insert/delete but allows exact matches to take precedence
- The `repeat_insert_delete_count` is reset for case-change operations, treating them like matches
- Random perturbation (`_bit_of_random()`) is applied to case-change costs when `randomize_edit_path=True`