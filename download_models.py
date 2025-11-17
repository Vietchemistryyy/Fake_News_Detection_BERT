"""
Download pre-trained models from Hugging Face
Run this script after cloning the repository
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_models():
    """Download fine-tuned models from Hugging Face"""
    print("="*80)
    print("DOWNLOADING MODELS FROM HUGGING FACE")
    print("="*80)
    print("\n📦 Downloading fine-tuned models for Fake News Detection...")
    print("   Models trained by: Nguyen Quoc Viet")
    print("   Accuracy: 92%+ for both English and Vietnamese")
    
    # Create directories
    models_dir = Path("models")
    bert_dir = models_dir / "BERT"
    phobert_dir = models_dir / "PhoBERT"
    
    bert_dir.mkdir(parents=True, exist_ok=True)
    phobert_dir.mkdir(parents=True, exist_ok=True)
    
    # Download RoBERTa (English) - Fine-tuned
    print("\n1️⃣ Downloading RoBERTa model for English...")
    print("   Source: Vietchemistryyy/fake-news-roberta-english")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Vietchemistryyy/fake-news-roberta-english")
        model = AutoModelForSequenceClassification.from_pretrained(
            "Vietchemistryyy/fake-news-roberta-english"
        )
        
        tokenizer.save_pretrained(bert_dir)
        model.save_pretrained(bert_dir)
        print("   ✅ RoBERTa downloaded successfully!")
        print("   📊 Performance: 92%+ accuracy on English fake news")
        print(f"   📁 Saved to: {bert_dir}")
    except Exception as e:
        print(f"   ❌ Error downloading RoBERTa: {e}")
        print("   💡 Falling back to base model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
            tokenizer.save_pretrained(bert_dir)
            model.save_pretrained(bert_dir)
            print("   ⚠️  Using base model (lower accuracy, needs fine-tuning)")
        except Exception as e2:
            print(f"   ❌ Failed: {e2}")
            return False
    
    # Download PhoBERT (Vietnamese) - Fine-tuned
    print("\n2️⃣ Downloading PhoBERT model for Vietnamese...")
    print("   Source: Vietchemistryyy/fake-news-phobert-vietnamese")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Vietchemistryyy/fake-news-phobert-vietnamese")
        model = AutoModelForSequenceClassification.from_pretrained(
            "Vietchemistryyy/fake-news-phobert-vietnamese"
        )
        
        tokenizer.save_pretrained(phobert_dir)
        model.save_pretrained(phobert_dir)
        print("   ✅ PhoBERT downloaded successfully!")
        print("   📊 Performance: 92%+ accuracy on Vietnamese fake news")
        print(f"   📁 Saved to: {phobert_dir}")
    except Exception as e:
        print(f"   ❌ Error downloading PhoBERT: {e}")
        print("   💡 Falling back to base model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)
            tokenizer.save_pretrained(phobert_dir)
            model.save_pretrained(phobert_dir)
            print("   ⚠️  Using base model (lower accuracy, needs fine-tuning)")
        except Exception as e2:
            print(f"   ❌ Failed: {e2}")
            return False
    
    print("\n" + "="*80)
    print("✅ MODEL DOWNLOAD COMPLETED")
    print("="*80)
    print("\n📊 Model Information:")
    print("   - RoBERTa (English): ~480 MB")
    print("   - PhoBERT (Vietnamese): ~517 MB")
    print("   - Total: ~1 GB")
    print("\n🚀 Next Steps:")
    print("   1. Start MongoDB: net start MongoDB")
    print("   2. Start backend: cd api && python main.py")
    print("   3. Start frontend: cd fe && npm run dev")
    print("   4. Open browser: http://localhost:3000")
    print("\n💡 Model Details:")
    print("   - Hugging Face: https://huggingface.co/Vietchemistryyy")
    print("   - RoBERTa: https://huggingface.co/Vietchemistryyy/fake-news-roberta-english")
    print("   - PhoBERT: https://huggingface.co/Vietchemistryyy/fake-news-phobert-vietnamese")
    
    return True

if __name__ == "__main__":
    try:
        success = download_models()
        if success:
            print("\n✨ Ready to detect fake news!")
        else:
            print("\n⚠️  Some models failed to download. Check errors above.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
