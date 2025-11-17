# Chain of Clarifications - Quick Reference Guide

## All 4 Critical Issues: RESOLVED ✅

---

## Issue 1: Answer Extraction - Key Functions

### BaseAgent.generate_response() - Response Validation
```python
# Empty check
if not response:
    return "[ERROR: Empty response generated]"

# Short response warning
if len(response) < 10:
    logger.warning(f"Very short response: {len(response)} chars")

# Prompt contamination check (CausalLM only)
if not self.is_seq2seq and response_start.startswith(prompt_start):
    return "[ERROR: Response contaminated with prompt text]"
```

### Verifier/Reasoner - Improved Answer Extraction
```python
# 1. Check for error markers first
if verification_output.startswith("[ERROR:"):
    return verification_output

# 2. Try pattern matching (Final Answer:, Answer:, etc.)
for pattern in patterns:
    match = re.search(pattern, output)
    if match and self._is_valid_answer(match.group(1)):
        return match.group(1)

# 3. Fallback: last sentence (not first!)
for sent in reversed(sentences):
    if self._is_valid_answer(sent):
        return sent
```

---

## Issue 2: Model Upgrade - Flan-T5-Base

### Default Model Changed
```python
# All agent classes now default to:
model_name: str = "google/flan-t5-base"

# Command line default:
parser.add_argument('--model_name', default='google/flan-t5-base')
```

### Dual Model Support
```python
# Auto-detects model type
is_t5_family = any(x in model_name.lower() for x in ['t5', 'flan'])

if is_t5_family:
    model = AutoModelForSeq2SeqLM.from_pretrained(...)
    is_seq2seq = True
else:
    model = AutoModelForCausalLM.from_pretrained(...)
    is_seq2seq = False

# Generation differs by type
if is_seq2seq:
    # T5: decode entire output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
else:
    # GPT: skip input tokens
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
```

---

## Issue 3: GPU Utilization - Diagnostics & Forcing

### GPU Diagnostics (First Agent Only)
```python
def _log_gpu_diagnostics(self):
    """Logs comprehensive GPU info"""
    - CUDA availability
    - GPU count and names
    - Total memory per GPU
    - CUDA capability version
    - Current memory allocation
    - CUDA/cuDNN versions
```

### Force CUDA Device
```python
if device == "cuda":
    cuda_device = torch.device("cuda:0")  # Force device 0
    model.to(cuda_device)
    
    # Log memory after loading
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    logger.info(f"GPU Memory Allocated: {allocated:.2f} MB")
```

### Error Handling with CPU Fallback
```python
try:
    model.to(cuda_device)
except RuntimeError as e:
    logger.error(f"Failed to move to GPU: {e}")
    logger.warning("Falling back to CPU")
    device = "cpu"
    model.to(device)
```

---

## Issue 4: Shared Model Architecture

### AgentChain - Load Once, Share Everywhere
```python
class AgentChain:
    def __init__(self, model_name, device, ...):
        # Load model ONCE
        self.shared_model, self.shared_tokenizer = self._load_shared_model()
        
        # Pass to all agents
        self.retriever = RetrieverAgent(
            model=self.shared_model,
            tokenizer=self.shared_tokenizer
        )
        self.reasoner = ReasonerAgent(
            model=self.shared_model,
            tokenizer=self.shared_tokenizer
        )
        self.verifier = VerifierAgent(
            model=self.shared_model,
            tokenizer=self.shared_tokenizer
        )
```

### BaseAgent - Accept Shared Models
```python
def __init__(self, role, model_name, 
             model=None,      # ✅ Optional pre-loaded model
             tokenizer=None): # ✅ Optional pre-loaded tokenizer
    
    if model is None or tokenizer is None:
        self._load_model()  # Load new
    else:
        self.model = model          # Use shared
        self.tokenizer = tokenizer  # Use shared
        # Detect type from class name
        self.is_seq2seq = 'Seq2Seq' in model.__class__.__name__
```

### Memory Savings
- **Before**: 3 model copies in GPU memory
- **After**: 1 model copy shared by all agents
- **Savings**: ~66% reduction in GPU memory usage

---

## File Locations

### Modified Files
1. `/home/hrsh/MEGA_PROJECTS/research_paper/agents/base_agent.py`
   - Lines 28-88: Shared model support
   - Lines 90-132: GPU diagnostics
   - Lines 134-201: Dual model loading
   - Lines 264-285: Response validation

2. `/home/hrsh/MEGA_PROJECTS/research_paper/agents/verifier.py`
   - Lines 148-226: Improved answer extraction

3. `/home/hrsh/MEGA_PROJECTS/research_paper/agents/reasoner.py`
   - Lines 144-222: Improved answer extraction

4. `/home/hrsh/MEGA_PROJECTS/research_paper/agents/agent_chain.py`
   - Lines 53-76: Agent initialization with shared model
   - Lines 85-157: Shared model loading method

5. `/home/hrsh/MEGA_PROJECTS/research_paper/experiments/baseline.py`
   - Line 335: Default model changed to flan-t5-base

---

## Running the Code

### Basic Experiment
```bash
python3 experiments/baseline.py \
    --num_examples 10 \
    --compression_type none
```

### With Custom Model (still supported!)
```bash
python3 experiments/baseline.py \
    --model_name gpt2 \
    --num_examples 5
```

### Full Comparison
```bash
python3 experiments/baseline.py \
    --comparison \
    --num_examples 100
```

---

## Testing the Fixes

### Test Issue 1: Answer Extraction
- Run with small model and check logs for validation warnings
- Look for "[ERROR: Empty response]" or "[ERROR: Response contaminated]"

### Test Issue 2: Model Support
- Default runs should use flan-t5-base
- Try `--model_name gpt2` to verify CausalLM still works
- Check logs for "Detected Seq2Seq model" vs "Detected CausalLM model"

### Test Issue 3: GPU Utilization
- Check first agent logs for GPU diagnostics section
- Look for "Moving model to cuda:0" and memory stats
- Monitor `nvidia-smi` to verify GPU usage

### Test Issue 4: Shared Model
- Check logs for "Loading shared model" (once)
- Each agent should log "Using shared pre-loaded model"
- Compare GPU memory: shared should use ~66% less than separate

---

## Key Improvements Summary

1. **Robustness**: Handles empty, short, and contaminated responses
2. **Flexibility**: Supports both Seq2Seq and CausalLM models
3. **Visibility**: Comprehensive GPU diagnostics for debugging
4. **Efficiency**: Shared model architecture reduces memory by 66%
5. **Quality**: All code compiles, follows best practices, fully documented

