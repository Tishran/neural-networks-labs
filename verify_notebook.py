"""
Comprehensive verification script for experiments.ipynb
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def verify_notebook():
    """Verify all requirements for the notebook."""

    # Load notebook
    notebook_path = Path('experiments.ipynb')
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    issues = []
    warnings = []

    print("=" * 80)
    print("COMPREHENSIVE NOTEBOOK VERIFICATION")
    print("=" * 80)

    # 1. Check imports in cell 3
    print("\n1. CHECKING IMPORTS (Cell 3)")
    print("-" * 80)
    cell_3 = cells[3]
    imports_text = ''.join(cell_3['source'])

    required_imports = [
        'torch', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tqdm', 'json',
        'pathlib', 'collections', 'random',
        'create_datasets', 'create_dataloaders', 'get_dataset_statistics',
        'create_model', 'Seq2Seq', 'Encoder', 'Decoder',
        'LuongAttention', 'Trainer', 'get_device', 'count_parameters',
        'calculate_metrics', 'levenshtein_distance', 'position_wise_accuracy',
        'decode_with_strategy', 'greedy_decode', 'beam_search',
        'plot_training_curves_comparison', 'plot_training_curves',
        'ExperimentConfig', 'get_baseline_config', 'get_experiment_configs',
        'get_decoding_configs'
    ]

    missing_imports = []
    for imp in required_imports:
        if imp not in imports_text:
            missing_imports.append(imp)

    if missing_imports:
        issues.append(f"Missing imports: {missing_imports}")
        print(f"[FAIL] Missing imports: {missing_imports}")
    else:
        print(f"[PASS] All required imports present")

    # 2. Check dataset configuration in cell 5
    print("\n2. CHECKING DATASET CONFIGURATION (Cell 5)")
    print("-" * 80)
    cell_5 = cells[5]
    dataset_text = ''.join(cell_5['source'])

    dataset_checks = {
        'max_num=250000': 'Extended range to 250000',
        'sample_size=80000': '80K total samples',
        'train_ratio=0.8': '80% train ratio',
        'val_ratio=0.1': '10% validation ratio',
        'test_ratio=0.1': '10% test ratio',
        'seed=42': 'Seed for reproducibility'
    }

    for check, desc in dataset_checks.items():
        if check in dataset_text:
            print(f"[OK] PASS: {desc} ({check})")
        else:
            issues.append(f"Dataset config missing: {desc}")
            print(f"[X] FAIL: {desc} ({check})")

    # 3. Check run_experiment function in cell 10
    print("\n3. CHECKING run_experiment FUNCTION (Cell 10)")
    print("-" * 80)
    cell_10 = cells[10]
    run_exp_text = ''.join(cell_10['source'])

    run_exp_checks = {
        'resume_from_checkpoint=True': 'Resume from checkpoint parameter',
        'show_examples=5': 'Show 5 examples parameter',
        'checkpoint_path.exists()': 'Checkpoint existence check',
        'history': 'History tracking'
    }

    for check, desc in run_exp_checks.items():
        if check in run_exp_text:
            print(f"[OK] PASS: {desc}")
        else:
            issues.append(f"run_experiment missing: {desc}")
            print(f"[X] FAIL: {desc}")

    # 4. Check all experiments store test_metrics AND history
    print("\n4. CHECKING EXPERIMENT RESULT STORAGE")
    print("-" * 80)

    experiment_cells = {
        'range_results': (14, 'Range experiments'),
        'hidden_results': (18, 'Hidden size experiments'),
        'layer_results': (22, 'Layer experiments'),
        'cell_results': (26, 'Cell type experiments'),
        'direction_results': (30, 'Direction experiments'),
        'embed_results': (34, 'Embedding experiments'),
        'attention_results': (38, 'Attention experiments'),
        'dropout_results': (42, 'Dropout experiments')
    }

    for var_name, (cell_idx, desc) in experiment_cells.items():
        cell = cells[cell_idx]
        cell_text = ''.join(cell['source'])

        # Check both test_metrics and history are stored
        has_test_metrics = "'test_metrics'" in cell_text or "['test_metrics']" in cell_text
        has_history = "'history'" in cell_text or "['history']" in cell_text

        if has_test_metrics and has_history:
            print(f"[OK] PASS: {desc} ({var_name}) stores test_metrics AND history")
        elif has_test_metrics:
            warnings.append(f"{desc} ({var_name}) stores test_metrics but MISSING history")
            print(f"[\!] WARN: {desc} ({var_name}) stores test_metrics but MISSING history")
        elif has_history:
            warnings.append(f"{desc} ({var_name}) stores history but MISSING test_metrics")
            print(f"[\!] WARN: {desc} ({var_name}) stores history but MISSING test_metrics")
        else:
            issues.append(f"{desc} ({var_name}) missing both test_metrics and history")
            print(f"[X] FAIL: {desc} ({var_name}) missing both test_metrics and history")

    # 5. Check all figures are saved and no duplicates
    print("\n5. CHECKING FIGURE SAVE PATHS")
    print("-" * 80)

    save_paths = []
    for i, cell in enumerate(cells):
        cell_text = ''.join(cell['source'])
        # Find all save_path occurrences
        import re
        matches = re.findall(r"save_path=['\"]([^'\"]+)['\"]", cell_text)
        for match in matches:
            save_paths.append((i, match))

    # Check for duplicates
    path_counts = defaultdict(list)
    for cell_idx, path in save_paths:
        path_counts[path].append(cell_idx)

    duplicates = {path: cells for path, cells in path_counts.items() if len(cells) > 1}

    print(f"Total figures to save: {len(save_paths)}")
    print("\nAll save paths:")
    for cell_idx, path in sorted(save_paths, key=lambda x: x[1]):
        print(f"  Cell {cell_idx}: {path}")

    if duplicates:
        issues.append(f"Duplicate save paths found: {duplicates}")
        print(f"\n[X] FAIL: Duplicate save paths:")
        for path, cell_indices in duplicates.items():
            print(f"  {path} in cells {cell_indices}")
    else:
        print(f"\n[OK] PASS: No duplicate save paths")

    # 6. Check training curves for all 8 experiment groups
    print("\n6. CHECKING TRAINING CURVES PLOTTING")
    print("-" * 80)

    training_curve_cells = [
        (16, 'range_results', 'Range'),
        (20, 'hidden_results', 'Hidden Size'),
        (24, 'layer_results', 'Layers'),
        (28, 'cell_results', 'Cell Type'),
        (32, 'direction_results', 'Direction'),
        (36, 'embed_results', 'Embedding'),
        (40, 'attention_results', 'Attention'),
        (44, 'dropout_results', 'Dropout')
    ]

    missing_curves = []
    for cell_idx, var_name, desc in training_curve_cells:
        cell = cells[cell_idx]
        cell_text = ''.join(cell['source'])

        has_loss_curve = 'loss' in cell_text.lower() and 'plot_training_curves_comparison' in cell_text
        has_acc_curve = 'accuracy' in cell_text.lower() and 'plot_training_curves_comparison' in cell_text

        if has_loss_curve and has_acc_curve:
            print(f"[OK] PASS: {desc} has both loss and accuracy curves")
        else:
            missing_type = []
            if not has_loss_curve:
                missing_type.append('loss')
            if not has_acc_curve:
                missing_type.append('accuracy')
            missing_curves.append(f"{desc} missing {missing_type} curves")
            print(f"[X] FAIL: {desc} missing {missing_type} curves")

    if missing_curves:
        issues.extend(missing_curves)

    # 7. Check attention visualization (3 separate figures)
    print("\n7. CHECKING ATTENTION VISUALIZATION (Cell 50)")
    print("-" * 80)
    cell_50 = cells[50]
    attn_text = ''.join(cell_50['source'])

    attn_checks = {
        'attention_heatmaps_small.png': 'Small numbers figure',
        'attention_heatmaps_medium.png': 'Medium numbers figure',
        'attention_heatmaps_large.png': 'Large numbers figure',
        'small_numbers': 'Small numbers examples',
        'medium_numbers': 'Medium numbers examples',
        'large_numbers': 'Large numbers examples'
    }

    for check, desc in attn_checks.items():
        if check in attn_text:
            print(f"[OK] PASS: {desc}")
        else:
            issues.append(f"Attention viz missing: {desc}")
            print(f"[X] FAIL: {desc}")

    # 8. Check decoding strategies (10 strategies)
    print("\n8. CHECKING DECODING STRATEGIES (Cell 47)")
    print("-" * 80)
    cell_47 = cells[47]
    decode_text = ''.join(cell_47['source'])

    if 'get_decoding_configs()' in decode_text:
        print(f"[OK] PASS: Uses get_decoding_configs() which returns 10 strategies")
        print("  Strategies: greedy, beam(3,5,10), top_k(3,5,5@0.7), top_p(0.9,0.95,0.9@0.7)")
    else:
        issues.append("Decoding strategies not using get_decoding_configs()")
        print(f"[X] FAIL: Not using get_decoding_configs()")

    # 9. Check error analysis (train and test confusion matrices)
    print("\n9. CHECKING ERROR ANALYSIS (Cell 56)")
    print("-" * 80)
    cell_56 = cells[56]
    error_text = ''.join(cell_56['source'])

    error_checks = {
        'test_conf_matrices': 'Test confusion matrices',
        'train_conf_matrices': 'Train confusion matrices',
        'confusion_matrices_test.png': 'Test confusion figure',
        'confusion_matrices_train.png': 'Train confusion figure',
        'position_wise_accuracy(test_preds': 'Test position accuracy',
        'position_wise_accuracy(train_preds': 'Train position accuracy'
    }

    for check, desc in error_checks.items():
        if check in error_text:
            print(f"[OK] PASS: {desc}")
        else:
            warnings.append(f"Error analysis might be missing: {desc}")
            print(f"[\!] WARN: {desc}")

    # 10. Check summary table (Cell 59)
    print("\n10. CHECKING SUMMARY TABLE (Cell 59)")
    print("-" * 80)
    cell_59 = cells[59]
    summary_text = ''.join(cell_59['source'])

    # Check all result variables are accessed
    result_vars = [
        'baseline_result', 'range_results', 'hidden_results', 'layer_results',
        'cell_results', 'direction_results', 'embed_results',
        'attention_results', 'dropout_results'
    ]

    for var in result_vars:
        if var in summary_text:
            # Check if test_metrics is accessed
            if "'test_metrics']" in summary_text or "['test_metrics']" in summary_text:
                print(f"[OK] PASS: {var} accessed with test_metrics")
            else:
                warnings.append(f"{var} accessed but test_metrics pattern unclear")
                print(f"[\!] WARN: {var} accessed but test_metrics pattern unclear")
        else:
            issues.append(f"Summary table missing: {var}")
            print(f"[X] FAIL: {var} not in summary table")

    # 11. Check for syntax errors
    print("\n11. CHECKING FOR SYNTAX ERRORS")
    print("-" * 80)

    # Look for common syntax errors
    syntax_patterns = [
        (r'\[\s*\]\.', 'Empty list indexing'),
        (r'\{\s*\}\.', 'Empty dict accessing'),
        (r'=\s*=\s*[^=]', 'Assignment with =='),
        (r'\)\s*\(', 'Missing operator between parens'),
    ]

    import re
    syntax_issues = []
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            cell_text = ''.join(cell['source'])
            for pattern, desc in syntax_patterns:
                if re.search(pattern, cell_text):
                    syntax_issues.append(f"Cell {i}: Potential {desc}")

    if syntax_issues:
        warnings.extend(syntax_issues)
        print(f"[\!] WARN: Potential syntax issues:")
        for issue in syntax_issues:
            print(f"  {issue}")
    else:
        print(f"[OK] PASS: No obvious syntax errors detected")

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if not issues and not warnings:
        print("[OK] ALL CHECKS PASSED!")
        return True
    else:
        if issues:
            print(f"\n[X] CRITICAL ISSUES ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        if warnings:
            print(f"\n[\!] WARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        return len(issues) == 0

if __name__ == '__main__':
    success = verify_notebook()
    exit(0 if success else 1)
