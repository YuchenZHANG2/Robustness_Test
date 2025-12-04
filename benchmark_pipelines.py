"""
Benchmark script to compare standard vs batch-optimized pipeline performance.
"""
import time
import torch
from model_loader import ModelLoader
from evaluator import COCOEvaluator
from testing_pipeline import RobustnessTest
from batch_optimized_pipeline import BatchOptimizedRobustnessTest

def benchmark_pipelines():
    """Compare performance of different pipeline implementations."""
    
    # Initialize components
    print("Initializing model loader and evaluator...")
    model_loader = ModelLoader()
    evaluator = COCOEvaluator(
        annotation_file='/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
        image_dir='/home/yuchen/YuchenZ/Datasets/coco/val2017'
    )
    
    # Test parameters
    model_keys = ['frcnn_v2']  # Single model for quick test
    corruption_names = ['gaussian_noise', 'motion_blur']  # Two corruptions
    image_ids = evaluator.get_random_image_ids(50)  # 50 images
    severities = [1, 3, 5]  # Three severity levels
    
    print(f"\nBenchmark Configuration:")
    print(f"  Models: {model_keys}")
    print(f"  Corruptions: {corruption_names}")
    print(f"  Images: {len(image_ids)}")
    print(f"  Severities: {severities}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Benchmark 1: Standard pipeline (sequential)
    print("\n" + "="*70)
    print("BENCHMARK 1: Standard Pipeline (Sequential)")
    print("="*70)
    
    test1 = RobustnessTest(model_loader, evaluator)
    start_time = time.time()
    
    results1 = test1.run_full_test(
        model_keys=model_keys,
        corruption_names=corruption_names,
        image_ids=image_ids,
        severities=severities
    )
    
    time1 = time.time() - start_time
    print(f"\n✓ Standard Pipeline completed in {time1:.2f} seconds")
    
    # Benchmark 2: Batch-optimized pipeline
    print("\n" + "="*70)
    print("BENCHMARK 2: Batch-Optimized Pipeline (Parallel + DataLoader)")
    print("="*70)
    
    test2 = BatchOptimizedRobustnessTest(
        model_loader, 
        evaluator,
        batch_size=8,
        num_workers=4
    )
    start_time = time.time()
    
    results2 = test2.run_full_test(
        model_keys=model_keys,
        corruption_names=corruption_names,
        image_ids=image_ids,
        severities=severities
    )
    
    time2 = time.time() - start_time
    print(f"\n✓ Batch-Optimized Pipeline completed in {time2:.2f} seconds")
    
    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Standard Pipeline:        {time1:.2f}s")
    print(f"Batch-Optimized Pipeline: {time2:.2f}s")
    print(f"Speedup:                  {time1/time2:.2f}x faster")
    print(f"Time Saved:               {time1-time2:.2f}s ({(time1-time2)/time1*100:.1f}%)")
    
    # Verify results match
    print("\n" + "="*70)
    print("ACCURACY VERIFICATION")
    print("="*70)
    
    for model_key in model_keys:
        clean_map1 = results1[model_key]['clean']['mAP']
        clean_map2 = results2[model_key]['clean']['mAP']
        
        print(f"Clean mAP comparison:")
        print(f"  Standard:  {clean_map1:.4f}")
        print(f"  Optimized: {clean_map2:.4f}")
        print(f"  Difference: {abs(clean_map1 - clean_map2):.6f}")
        
        if abs(clean_map1 - clean_map2) < 0.001:
            print("  ✓ Results match!")
        else:
            print("  ⚠ Results differ (may be due to numerical precision)")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROBUSTNESS TESTING PIPELINE BENCHMARK")
    print("="*70)
    
    benchmark_pipelines()
    
    print("\n" + "="*70)
    print("Benchmark completed!")
    print("="*70)
