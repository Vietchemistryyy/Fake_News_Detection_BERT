"""
Verification script to ensure entire codebase is RoBERTa-ready
Run this before training to verify setup
"""

import sys
sys.path.append('..')

def verify_roberta_setup():
    """Comprehensive verification of RoBERTa setup"""
    
    print("="*80)
    print("🔍 ROBERTA SETUP VERIFICATION")
    print("="*80)
    
    issues = []
    warnings = []
    
    # 1. Check config
    print("\n📋 Checking configuration...")
    try:
        from src.config import ModelConfig
        
        if ModelConfig.MODEL_NAME == "roberta-base":
            print(f"   ✅ Model: {ModelConfig.MODEL_NAME}")
        else:
            issues.append(f"Config uses {ModelConfig.MODEL_NAME} instead of roberta-base")
            print(f"   ❌ Model: {ModelConfig.MODEL_NAME} (Expected: roberta-base)")
        
        if ModelConfig.MAX_LENGTH == 256:
            print(f"   ✅ Max length: {ModelConfig.MAX_LENGTH}")
        else:
            warnings.append(f"MAX_LENGTH is {ModelConfig.MAX_LENGTH}, consider 256 for RoBERTa")
            print(f"   ⚠️  Max length: {ModelConfig.MAX_LENGTH}")
        
        if ModelConfig.BATCH_SIZE == 16:
            print(f"   ✅ Batch size: {ModelConfig.BATCH_SIZE}")
        else:
            warnings.append(f"BATCH_SIZE is {ModelConfig.BATCH_SIZE}, RoBERTa typically uses 16")
            print(f"   ⚠️  Batch size: {ModelConfig.BATCH_SIZE}")
            
    except Exception as e:
        issues.append(f"Config check failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # 2. Check model.py for DistilBERT references
    print("\n🤖 Checking model.py...")
    try:
        with open('../src/model.py', 'r') as f:
            content = f.read()
        
        if 'DistilBertForSequenceClassification' in content:
            issues.append("model.py still has DistilBertForSequenceClassification imports")
            print("   ❌ Found DistilBert hardcoded references")
        else:
            print("   ✅ No DistilBert hardcoded references")
        
        if 'AutoModelForSequenceClassification' in content:
            print("   ✅ Using Auto classes (generic)")
        else:
            issues.append("model.py doesn't use AutoModelForSequenceClassification")
            print("   ❌ Not using Auto classes")
            
    except Exception as e:
        warnings.append(f"Could not check model.py: {e}")
        print(f"   ⚠️  Could not verify: {e}")
    
    # 3. Check transformers availability
    print("\n📦 Checking transformers...")
    try:
        import transformers
        print(f"   ✅ Transformers version: {transformers.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("   ✅ Auto classes available")
        
        # Try loading RoBERTa tokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        print("   ✅ RoBERTa tokenizer loaded successfully")
        print(f"   ✅ Vocab size: {tokenizer.vocab_size:,}")
        
    except ImportError as e:
        issues.append(f"Transformers not available: {e}")
        print(f"   ❌ Transformers not available: {e}")
    except Exception as e:
        issues.append(f"Error loading RoBERTa: {e}")
        print(f"   ❌ Error: {e}")
    
    # 4. Check train.py
    print("\n🚀 Checking train.py...")
    try:
        from src.train import BertTrainer, TRANSFORMERS_AVAILABLE, EARLY_STOPPING_AVAILABLE
        
        print(f"   ✅ BertTrainer imported")
        print(f"   ✅ TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
        print(f"   ✅ EARLY_STOPPING_AVAILABLE: {EARLY_STOPPING_AVAILABLE}")
        
        if not TRANSFORMERS_AVAILABLE:
            issues.append("Transformers not available in train.py")
        
    except Exception as e:
        issues.append(f"train.py check failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # 5. Check evaluate.py
    print("\n📊 Checking evaluate.py...")
    try:
        from src.evaluate import compute_extended_metrics, compute_metrics
        
        print("   ✅ compute_extended_metrics available")
        print("   ✅ compute_metrics available")
        
    except Exception as e:
        issues.append(f"evaluate.py check failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # 6. Check dataset.py
    print("\n📂 Checking dataset.py...")
    try:
        from src.dataset import create_dataset_from_dataframe
        
        print("   ✅ create_dataset_from_dataframe available")
        
    except Exception as e:
        issues.append(f"dataset.py check failed: {e}")
        print(f"   ❌ Error: {e}")
    
    # 7. Check data files
    print("\n💾 Checking data files...")
    try:
        from src.config import DataConfig
        import os
        
        files_to_check = [
            ('Train', DataConfig.TRAIN_PATH),
            ('Val', DataConfig.VAL_PATH),
            ('Test', DataConfig.TEST_PATH)
        ]
        
        for name, path in files_to_check:
            if os.path.exists(path):
                print(f"   ✅ {name}: {path}")
            else:
                warnings.append(f"{name} data not found at {path}")
                print(f"   ⚠️  {name}: Not found")
                
    except Exception as e:
        warnings.append(f"Could not check data files: {e}")
        print(f"   ⚠️  Error: {e}")
    
    # 8. Test model creation
    print("\n🧪 Testing model creation...")
    try:
        from src.model import create_bert_model
        
        model_wrapper = create_bert_model("roberta-base")
        print("   ✅ RoBERTa model wrapper created")
        
        # Try loading (this will download if not cached)
        print("   📥 Loading RoBERTa (may take a moment)...")
        model_wrapper.load_model()
        print(f"   ✅ Model type: {type(model_wrapper.model).__name__}")
        print(f"   ✅ Tokenizer type: {type(model_wrapper.tokenizer).__name__}")
        
    except Exception as e:
        warnings.append(f"Model creation test failed: {e}")
        print(f"   ⚠️  Test failed: {e}")
    
    # Final report
    print("\n" + "="*80)
    print("📋 VERIFICATION REPORT")
    print("="*80)
    
    if not issues and not warnings:
        print("\n✅ ALL CHECKS PASSED!")
        print("🎉 Codebase is fully RoBERTa-ready!")
        print("\n💡 You can proceed with training:")
        print("   jupyter notebook notebooks/04_bert_training.ipynb")
        return True
    
    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    if issues:
        print("\n🔧 Please fix critical issues before training.")
        return False
    else:
        print("\n✅ No critical issues. Warnings are optional.")
        print("💡 You can proceed with training.")
        return True


if __name__ == "__main__":
    success = verify_roberta_setup()
    sys.exit(0 if success else 1)