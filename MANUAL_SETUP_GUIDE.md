# Manual Setup Guide: Chain of Clarifications Project

## Hardware Requirements
- **GPU**: RTX 4050 or equivalent
- **VRAM**: 6GB minimum
- **RAM**: 16GB recommended
- **Storage**: 10GB free space for models and datasets

## System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), Windows 10/11, or macOS
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.7+ (for GPU acceleration)

---

## Task 1: Python Environment Setup

### 1.1 Create Virtual Environment
```bash
# Using venv (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Alternative: Using conda
conda create -n chain-clarify python=3.10
conda activate chain-clarify
```

### 1.2 Verify Python Version
```bash
python --version
# Should show Python 3.8.x, 3.9.x, or 3.10.x
```

---

## Task 2: Install PyTorch

### 2.1 Install PyTorch with CUDA Support

**For CUDA 11.8** (recommended for RTX 4050):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only** (not recommended for this project):
```bash
pip install torch torchvision torchaudio
```

### 2.2 Verify PyTorch Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output**:
```
PyTorch version: 2.1.0+cu118 (or similar)
CUDA available: True
CUDA version: 11.8 (or your installed version)
```

### 2.3 Check GPU Memory
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
```

**Expected Output**:
```
GPU: NVIDIA GeForce RTX 4050
VRAM: 6.00 GB
```

---

## Task 3: Install Transformers Library

### 3.1 Install Hugging Face Transformers
```bash
pip install transformers>=4.35.0
```

### 3.2 Install Additional Dependencies
```bash
pip install accelerate>=0.25.0
pip install sentencepiece
pip install protobuf
```

### 3.3 Verify Transformers Installation
```bash
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

**Expected Output**:
```
Transformers version: 4.35.0 (or higher)
```

---

## Task 4: Install Datasets Library

### 4.1 Install Datasets
```bash
pip install datasets>=2.15.0
```

### 4.2 Install Additional Data Processing Libraries
```bash
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
```

### 4.3 Verify Datasets Installation
```bash
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
```

---

## Task 5: Download and Test GPT-2 (124M params)

### 5.1 Create Test Script
Create a file named `test_gpt2.py`:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

def test_gpt2():
    print("=" * 60)
    print("GPT-2 124M Parameter Model Test")
    print("=" * 60)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load model and tokenizer
    print("\nLoading GPT-2 model and tokenizer...")
    start_time = time.time()

    model_name = "gpt2"  # 124M parameters
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    # Check model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Check memory usage
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Test inference
        print("\nTesting inference...")
        test_prompt = "The multi-agent system uses"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {test_prompt}")
        print(f"Generated: {generated_text}")

        # Memory stats
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory used: {peak_memory:.2f} GB")
        print(f"Remaining VRAM: {(torch.cuda.get_device_properties(0).total_memory / 1024**3) - peak_memory:.2f} GB")

        if peak_memory < 4.0:
            print("\n✓ SUCCESS: GPT-2 fits comfortably in 6GB VRAM!")
        else:
            print("\n⚠ WARNING: High memory usage. Consider using fp16 or quantization.")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_gpt2()
```

### 5.2 Run the Test
```bash
python test_gpt2.py
```

**Expected Output**:
```
============================================================
GPT-2 124M Parameter Model Test
============================================================

Device: cuda
GPU: NVIDIA GeForce RTX 4050
VRAM Available: 6.00 GB

Loading GPT-2 model and tokenizer...
Model loaded in 3.45 seconds
Total parameters: 124,439,808 (124.4M)

Testing inference...

Prompt: The multi-agent system uses
Generated: The multi-agent system uses role-specific compression...

Peak GPU memory used: 1.23 GB
Remaining VRAM: 4.77 GB

✓ SUCCESS: GPT-2 fits comfortably in 6GB VRAM!

============================================================
Test completed successfully!
============================================================
```

### 5.3 Optional: Test DistilGPT-2 (Smaller Alternative)
```python
# If GPT-2 uses too much memory, test DistilGPT-2
model_name = "distilgpt2"  # 82M parameters
```

---

## Task 6: Download SQuAD 1.1 Dataset

### 6.1 Create Dataset Download Script
Create a file named `download_squad.py`:

```python
from datasets import load_dataset
import os

def download_squad():
    print("=" * 60)
    print("Downloading SQuAD 1.1 Dataset")
    print("=" * 60)

    # Create data directory
    os.makedirs("data/squad", exist_ok=True)

    # Download SQuAD v1.1
    print("\nDownloading SQuAD v1.1 train split...")
    train_dataset = load_dataset("squad", split="train")
    print(f"Train examples: {len(train_dataset)}")

    print("\nDownloading SQuAD v1.1 validation split...")
    val_dataset = load_dataset("squad", split="validation")
    print(f"Validation examples: {len(val_dataset)}")

    # Display sample
    print("\nSample from dataset:")
    print("-" * 60)
    sample = train_dataset[0]
    print(f"Question: {sample['question']}")
    print(f"Context: {sample['context'][:200]}...")
    print(f"Answer: {sample['answers']['text'][0]}")
    print("-" * 60)

    # Save to disk (optional)
    print("\nSaving dataset to disk...")
    train_dataset.save_to_disk("data/squad/train")
    val_dataset.save_to_disk("data/squad/validation")

    print("\n✓ SQuAD 1.1 dataset downloaded successfully!")
    print(f"  - Train: {len(train_dataset)} examples")
    print(f"  - Validation: {len(val_dataset)} examples")
    print(f"  - Saved to: data/squad/")
    print("=" * 60)

if __name__ == "__main__":
    download_squad()
```

### 6.2 Run the Download Script
```bash
python download_squad.py
```

**Expected Output**:
```
============================================================
Downloading SQuAD 1.1 Dataset
============================================================

Downloading SQuAD v1.1 train split...
Train examples: 87599

Downloading SQuAD v1.1 validation split...
Validation examples: 10570

Sample from dataset:
------------------------------------------------------------
Question: When did Beyonce start becoming popular?
Context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say)...
Answer: in the late 1990s
------------------------------------------------------------

Saving dataset to disk...

✓ SQuAD 1.1 dataset downloaded successfully!
  - Train: 87599 examples
  - Validation: 10570 examples
  - Saved to: data/squad/
============================================================
```

---

## Task 7: Implement Basic Agent Class Structure

### 7.1 Create Base Agent Class
The base agent class is already in `agents/base_agent.py`. Here's the recommended implementation:

```python
# agents/base_agent.py
import torch
from typing import Dict, Any, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BaseAgent:
    """
    Base class for all agents in the multi-agent chain.
    Provides common functionality for context processing and model inference.
    """

    def __init__(
        self,
        role: str,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the base agent.

        Args:
            role: Agent role (e.g., "retriever", "reasoner", "verifier")
            model_name: Hugging Face model identifier
            device: Device to run the model on
        """
        self.role = role
        self.device = device
        self.model_name = model_name

        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()  # Set to evaluation mode

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the context and return updated context.
        This method should be overridden by subclasses.

        Args:
            context: Dictionary containing task information

        Returns:
            Updated context dictionary
        """
        raise NotImplementedError("Subclasses must implement process()")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using the agent's model.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage in GB.

        Returns:
            Memory usage in GB
        """
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def __repr__(self):
        return f"{self.__class__.__name__}(role='{self.role}', model='{self.model_name}')"
```

### 7.2 Verify Agent Structure
Create `test_agents.py`:

```python
# test_agents.py
import sys
sys.path.append('.')

from agents.base_agent import BaseAgent

def test_base_agent():
    print("Testing Base Agent...")
    agent = BaseAgent(role="test", model_name="gpt2")
    print(f"Agent created: {agent}")
    print(f"Device: {agent.device}")
    print(f"Memory usage: {agent.get_memory_usage():.2f} GB")

    # Test text generation
    prompt = "Multi-agent systems are"
    generated = agent.generate_text(prompt, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print("\n✓ Base agent test passed!")

if __name__ == "__main__":
    test_base_agent()
```

---

## Task 8: Create 3-Agent Chain (Retriever → Reasoner → Verifier)

### 8.1 Implement Retriever Agent
Update `agents/retriever.py`:

```python
# agents/retriever.py
from agents.base_agent import BaseAgent
from typing import Dict, Any

class RetrieverAgent(BaseAgent):
    """
    Retriever agent: Extracts relevant information from context.
    Focus: Find relevant passages, identify key entities, score relevance.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        super().__init__(role="retriever", model_name=model_name, device=device)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and retrieve relevant information.

        Args:
            context: {"question": str, "context": str}

        Returns:
            Updated context with retrieved passages
        """
        question = context.get("question", "")
        full_context = context.get("context", "")

        # Create retrieval prompt
        prompt = f"Question: {question}\nContext: {full_context[:500]}...\nRelevant information:"

        # Generate retrieval
        retrieved = self.generate_text(prompt, max_length=150)

        # Update context
        context["retrieved_passages"] = retrieved
        context["retriever_output"] = {
            "relevant_info": retrieved,
            "memory_used": self.get_memory_usage()
        }

        return context
```

### 8.2 Implement Reasoner Agent
Update `agents/reasoner.py`:

```python
# agents/reasoner.py
from agents.base_agent import BaseAgent
from typing import Dict, Any

class ReasonerAgent(BaseAgent):
    """
    Reasoner agent: Performs logical reasoning on retrieved information.
    Focus: Generate reasoning chains, identify candidate answers.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        super().__init__(role="reasoner", model_name=model_name, device=device)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process retrieved information and generate reasoning.

        Args:
            context: Contains question and retrieved passages

        Returns:
            Updated context with reasoning and candidate answer
        """
        question = context.get("question", "")
        retrieved = context.get("retrieved_passages", "")

        # Create reasoning prompt
        prompt = f"Question: {question}\nInformation: {retrieved}\nReasoning:"

        # Generate reasoning
        reasoning = self.generate_text(prompt, max_length=200)

        # Update context
        context["reasoning_chain"] = reasoning
        context["reasoner_output"] = {
            "reasoning": reasoning,
            "memory_used": self.get_memory_usage()
        }

        return context
```

### 8.3 Implement Verifier Agent
Update `agents/verifier.py`:

```python
# agents/verifier.py
from agents.base_agent import BaseAgent
from typing import Dict, Any

class VerifierAgent(BaseAgent):
    """
    Verifier agent: Verifies reasoning and produces final answer.
    Focus: Validate reasoning, extract final answer, provide confidence.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        super().__init__(role="verifier", model_name=model_name, device=device)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify reasoning and produce final answer.

        Args:
            context: Contains question, retrieved info, and reasoning

        Returns:
            Updated context with verified answer
        """
        question = context.get("question", "")
        reasoning = context.get("reasoning_chain", "")

        # Create verification prompt
        prompt = f"Question: {question}\nReasoning: {reasoning}\nFinal Answer:"

        # Generate final answer
        answer = self.generate_text(prompt, max_length=50)

        # Update context
        context["final_answer"] = answer
        context["verifier_output"] = {
            "answer": answer,
            "memory_used": self.get_memory_usage()
        }

        return context
```

### 8.4 Test the 3-Agent Chain
Create `test_chain.py`:

```python
# test_chain.py
import sys
sys.path.append('.')

from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.verifier import VerifierAgent

def test_agent_chain():
    print("=" * 60)
    print("Testing 3-Agent Chain: Retriever → Reasoner → Verifier")
    print("=" * 60)

    # Initialize agents
    print("\nInitializing agents...")
    retriever = RetrieverAgent()
    reasoner = ReasonerAgent()
    verifier = VerifierAgent()
    print(f"✓ {retriever}")
    print(f"✓ {reasoner}")
    print(f"✓ {verifier}")

    # Test input
    context = {
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France. It has an area of 105 square kilometres."
    }

    # Run chain
    print("\n" + "-" * 60)
    print("Running agent chain...")
    print("-" * 60)

    print("\n1. Retriever Agent:")
    context = retriever.process(context)
    print(f"   Retrieved: {context.get('retrieved_passages', '')[:100]}...")

    print("\n2. Reasoner Agent:")
    context = reasoner.process(context)
    print(f"   Reasoning: {context.get('reasoning_chain', '')[:100]}...")

    print("\n3. Verifier Agent:")
    context = verifier.process(context)
    print(f"   Answer: {context.get('final_answer', '')}")

    print("\n" + "=" * 60)
    print("✓ Agent chain test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_agent_chain()
```

---

## Installation Checklist

Use this checklist to track your progress:

### Environment Setup
- [ ] Python 3.8+ installed and verified
- [ ] Virtual environment created and activated
- [ ] GPU drivers installed (NVIDIA)
- [ ] CUDA toolkit installed

### Core Dependencies
- [ ] PyTorch installed with CUDA support
- [ ] PyTorch GPU functionality verified
- [ ] GPU memory checked (6GB available)
- [ ] Transformers library installed
- [ ] Datasets library installed
- [ ] Additional dependencies installed (pandas, numpy, scikit-learn)

### Models & Data
- [ ] GPT-2 (124M) downloaded and tested
- [ ] GPT-2 inference successful on GPU
- [ ] Memory usage verified (<2GB for GPT-2)
- [ ] SQuAD 1.1 dataset downloaded
- [ ] Dataset samples verified

### Agent Implementation
- [ ] Base agent class implemented
- [ ] Retriever agent implemented
- [ ] Reasoner agent implemented
- [ ] Verifier agent implemented
- [ ] 3-agent chain tested successfully

### Additional Setup
- [ ] Project directory structure verified
- [ ] Test scripts created and run
- [ ] Git repository initialized
- [ ] Dependencies documented in requirements.txt

---

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Error
```python
# Use smaller batch size or reduce context length
# Enable gradient checkpointing
# Use fp16 precision
model.half()  # Convert to half precision
```

### Slow Download Speeds
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Use mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Next Steps

After completing this manual setup:

1. **Verify Installation**: Run all test scripts
2. **Baseline Experiments**: Run `experiments/baseline.py`
3. **Track Memory**: Monitor VRAM usage during experiments
4. **Optimize**: Adjust batch sizes and context lengths as needed
5. **Document**: Keep notes on any issues and solutions

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate

# Check GPU status
nvidia-smi

# Test PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# Download model
python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"

# Run baseline experiment
python experiments/baseline.py --num_examples 10

# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

---

**Document Version**: 1.0
**Last Updated**: November 14, 2025
**Status**: Ready for Setup
