#!/usr/bin/env python3
"""
Full integration test for case-change feature including training and inference.
"""

import sys
sys.path.insert(0, 'src')

from sentence_transmorgrifier.transmorgrify import Transmorgrifier
import pandas as pd

def test_full_integration():
    """Test training and inference with case changes"""
    
    print("=" * 60)
    print("Full Integration Test: Case-Change Feature")
    print("=" * 60)
    print()
    
    # Create training data with various case transformations
    training_data = {
        'source': [
            'hello world',
            'the quick brown fox',
            'python programming',
            'machine learning',
            'artificial intelligence',
        ],
        'target': [
            'HELLO WORLD',
            'THE QUICK BROWN FOX',
            'PYTHON PROGRAMMING',
            'MACHINE LEARNING',
            'ARTIFICIAL INTELLIGENCE',
        ]
    }
    
    print("Training data (lowercase -> UPPERCASE):")
    for src, tgt in zip(training_data['source'], training_data['target']):
        print(f"  '{src}' -> '{tgt}'")
    print()
    
    # Create and train the model
    print("Training model...")
    tm = Transmorgrifier()
    tm.train(
        from_sentences=training_data['source'],
        to_sentences=training_data['target'],
        iterations=100,  # Small number for quick test
        device='cpu',
        trailing_context=7,
        leading_context=7,
        verbose=False
    )
    print("Training complete!")
    print()
    
    # Test inference on new data
    test_inputs = [
        'new test case',
        'another example',
        'case change test',
    ]
    
    print("Testing inference on new inputs:")
    results = list(tm.execute(test_inputs, verbose=False))
    
    for inp, out in zip(test_inputs, results):
        print(f"  Input:  '{inp}'")
        print(f"  Output: '{out}'")
        print(f"  Expected: '{inp.upper()}'")
        print(f"  Match: {out == inp.upper()}")
        print()
    
    # Test reverse transformation (UPPERCASE -> lowercase)
    print("-" * 60)
    print("Testing reverse transformation (UPPERCASE -> lowercase):")
    print()
    
    reverse_training = {
        'source': [
            'HELLO WORLD',
            'THE QUICK BROWN FOX',
            'PYTHON PROGRAMMING',
        ],
        'target': [
            'hello world',
            'the quick brown fox',
            'python programming',
        ]
    }
    
    tm2 = Transmorgrifier()
    tm2.train(
        from_sentences=reverse_training['source'],
        to_sentences=reverse_training['target'],
        iterations=100,
        device='cpu',
        verbose=False
    )
    
    test_inputs_upper = ['NEW TEST', 'ANOTHER EXAMPLE']
    results_lower = list(tm2.execute(test_inputs_upper, verbose=False))
    
    for inp, out in zip(test_inputs_upper, results_lower):
        print(f"  Input:  '{inp}'")
        print(f"  Output: '{out}'")
        print(f"  Expected: '{inp.lower()}'")
        print(f"  Match: {out == inp.lower()}")
        print()
    
    # Test title case (capitalize first letter of each word)
    print("-" * 60)
    print("Testing title case transformation:")
    print()
    
    title_training = {
        'source': [
            'hello world',
            'the quick brown fox',
            'python programming language',
            'machine learning algorithms',
            'data science projects',
        ],
        'target': [
            'Hello World',
            'The Quick Brown Fox',
            'Python Programming Language',
            'Machine Learning Algorithms',
            'Data Science Projects',
        ]
    }
    
    print("Training data (lowercase -> Title Case):")
    for src, tgt in zip(title_training['source'], title_training['target']):
        print(f"  '{src}' -> '{tgt}'")
    print()
    
    tm3 = Transmorgrifier()
    tm3.train(
        from_sentences=title_training['source'],
        to_sentences=title_training['target'],
        iterations=200,  # More iterations for this complex pattern
        device='cpu',
        verbose=False
    )
    
    test_inputs_title = [
        'new test case',
        'another example here',
        'artificial intelligence',
    ]
    
    print("Testing title case inference:")
    results_title = list(tm3.execute(test_inputs_title, verbose=False))
    
    for inp, out in zip(test_inputs_title, results_title):
        expected = inp.title()
        print(f"  Input:    '{inp}'")
        print(f"  Output:   '{out}'")
        print(f"  Expected: '{expected}'")
        print(f"  Match: {out == expected}")
        print()
    
    print("=" * 60)
    print("Full integration test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_integration()