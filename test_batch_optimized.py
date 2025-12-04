"""
Quick test of the batch-optimized pipeline to verify it works correctly.
"""
import torch
from model_loader import ModelLoader
from evaluator import COCOEvaluator
from batch_optimized_pipeline import BatchOptimizedRobustnessTest

def quick_test():
    """Run a quick test with minimal images to verify functionality."""
    
    print("="*70)
    print("QUICK TEST: Batch-Optimized Pipeline")
    print("="*70)
    
    # Initialize
    print("\n1. Initializing components...")
    model_loader = ModelLoader()
    evaluator = COCOEvaluator(
        annotation_file='/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
        image_dir='/home/yuchen/YuchenZ/Datasets/coco/val2017'
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   ✓ Using device: {device}")
    
    # Create tester
    print("\n2. Creating batch-optimized tester...")
    test = BatchOptimizedRobustnessTest(
        model_loader,
        evaluator,
        batch_size=4,  # Single GPU conservative batch size
        num_workers=2  # Parallel workers for loading
    )
    print("   ✓ Tester initialized")
    
    # Test with small dataset
    print("\n3. Running test with 10 images...")
    model_keys = ['frcnn_v2']
    corruption_names = ['gaussian_noise']
    image_ids = evaluator.get_random_image_ids(10)
    severities = [1, 3]
    
    print(f"   Models: {model_keys}")
    print(f"   Corruptions: {corruption_names}")
    print(f"   Images: {len(image_ids)}")
    print(f"   Severities: {severities}")
    
    # Run test
    results = test.run_full_test(
        model_keys=model_keys,
        corruption_names=corruption_names,
        image_ids=image_ids,
        severities=severities
    )
    
    # Display results
    print("\n4. Results:")
    print("="*70)
    for model_key, model_results in results.items():
        print(f"\nModel: {model_results['name']}")
        print(f"  Clean mAP: {model_results['clean']['mAP']:.4f}")
        
        for corruption, severity_results in model_results['corrupted'].items():
            print(f"  {corruption}:")
            for severity, metrics in severity_results.items():
                print(f"    Severity {severity}: mAP = {metrics['mAP']:.4f}")
    
    print("\n" + "="*70)
    print("✓ Quick test completed successfully!")
    print("="*70)

if __name__ == "__main__":
    quick_test()
