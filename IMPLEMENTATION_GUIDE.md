# Implementation Guide

## Complete Implementation of Chain of Clarifications Research Project

This document provides a comprehensive guide to the implemented system following the Revised Empirical Research Plan.

## ✅ Completed Implementation

### Phase 1: Foundation & Baseline (Weeks 1-3)

#### Week 1: Environment Setup & Agent Architecture ✅

**Implemented Components:**

1. **Base Agent Architecture** (`agents/base_agent.py`)
   - Generic agent class with LLM integration
   - GPU memory management
   - Token counting and tracking
   - Configurable generation parameters

2. **Three Specialized Agents:**
   - **Retriever** (`agents/retriever.py`): Extracts relevant information from documents
   - **Reasoner** (`agents/reasoner.py`): Generates answers with reasoning
   - **Verifier** (`agents/verifier.py`): Validates and refines answers

3. **Agent Chain Orchestrator** (`agents/agent_chain.py`)
   - Coordinates multi-agent pipeline
   - Handles context compression between agents
   - Tracks metrics and memory usage

4. **Utilities:**
   - **SQuAD Loader** (`data/load_squad.py`): Dataset loading and preprocessing
   - **Metrics Tracker** (`utils/metrics.py`): F1, EM, context sizes, latency
   - **Memory Tracker** (`utils/memory_tracker.py`): GPU/RAM monitoring

#### Week 2: Baseline Without Compression ✅

**Implementation:**
- Baseline experiment runner with comprehensive metrics
- Context size tracking at each agent transition
- Memory usage monitoring
- Latency measurement per example

**Run Command:**
```bash
python experiments/baseline.py --num_examples 100 --compression_type none
```

#### Week 3: Fixed Compression Baseline ✅

**Implemented Strategies:**

1. **NaiveCompressor** (`compression/naive_compression.py`):
   - `first_n`: Keep first N% of tokens
   - `last_n`: Keep last N% of tokens
   - `random`: Random sampling
   - `sentence_first`: Keep first N% of sentences

2. **SentenceScorer**: Importance scoring by:
   - Position
   - Length
   - Keyword overlap
   - Entity presence

**Run Command:**
```bash
python experiments/baseline.py --num_examples 100 --compression_type fixed --compression_ratio 0.5
```

### Phase 2: Clarification Mechanism (Weeks 4-6)

#### Week 4: Role-Specific Strategy Design ✅

**Implemented Components:**

1. **RoleSpecificScorer** (`compression/role_specific.py`)
   - Scores content based on next agent's needs
   - Different strategies for Reasoner vs Verifier

2. **Retriever → Reasoner Strategy:**
   - Prioritizes: question keywords, entities, facts
   - Downweights: background info, redundant content
   - Weights:
     - Keyword overlap: 3.0
     - Entity presence: 2.5
     - Position: 1.0
     - Length: 0.5

3. **Reasoner → Verifier Strategy:**
   - Prioritizes: final answer, reasoning chain, evidence
   - Downweights: exploratory reasoning, rejected options
   - Weights:
     - Answer indicators: 5.0
     - Reasoning markers: 3.0
     - Entities: 2.0
     - Reverse position: 1.5

#### Week 5: Adaptive Clarification Implementation ✅

**Implemented:**

1. **Clarifier Module** (`compression/role_specific.py`)
   - Adaptive compression based on importance distribution
   - Adjusts compression ratio dynamically:
     - High importance (>60%): Compress less (ratio × 1.3)
     - Low importance (<30%): Compress more (ratio × 0.7)
     - Medium: Use target ratio

2. **Integration:**
   - Seamless integration in `AgentChain`
   - Metadata passing between agents
   - Compression statistics tracking

**Run Command:**
```bash
python experiments/baseline.py --num_examples 100 --compression_type role_specific --compression_ratio 0.5
```

#### Week 6: Validation Experiments ✅

**Implemented:**

1. **Comprehensive Comparison** (`experiments/baseline.py`)
   - Supports multiple configurations
   - Statistical analysis built-in
   - Exports results to JSON

2. **Comparison Runner:**
   ```bash
   python experiments/baseline.py --num_examples 500 --comparison
   ```

   Tests:
   - No compression
   - Fixed: 25%, 50%, 75%
   - Role-specific: 25%, 50%, 75%

### Phase 3: Analysis Tools (Weeks 7-9)

#### Analysis and Visualization ✅

**Implemented Components:**

1. **ResultsAnalyzer** (`experiments/analyze_results.py`)
   - Loads all experimental results
   - Creates comparison DataFrames
   - Statistical testing (paired t-test, Cohen's d)
   - Confidence intervals

2. **Visualizations:**
   - F1 score comparison charts
   - Compression vs accuracy tradeoff
   - Context size reduction graphs

3. **Statistical Comparison:**
   ```bash
   python experiments/analyze_results.py --compare method1 method2
   ```

4. **Report Generation:**
   ```bash
   python experiments/analyze_results.py --report
   ```

## 🎯 How to Execute the Research Plan

### Step 1: Setup Environment

```bash
# Clone repository
git clone https://github.com/harshaygadekar/Chain-of-Clarifications.git
cd Chain-of-Clarifications

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Week 1 Checkpoint

Test the 3-agent system with 10 examples:

```bash
python experiments/baseline.py --num_examples 10 --compression_type none
```

**Expected Output:**
- All agents process examples successfully
- Context sizes logged at each transition
- Memory usage tracked
- Results saved to `results/` directory

### Step 3: Run Baseline Experiments (Week 2)

Run 100 examples without compression:

```bash
python experiments/baseline.py --num_examples 100 --compression_type none
```

**Metrics to Observe:**
- Baseline F1 score (target: >70%)
- Context explosion pattern
- Memory usage peaks
- Latency per example

### Step 4: Fixed Compression Experiments (Week 3)

Test multiple compression ratios:

```bash
# 25% compression
python experiments/baseline.py --num_examples 100 --compression_type fixed --compression_ratio 0.25

# 50% compression
python experiments/baseline.py --num_examples 100 --compression_type fixed --compression_ratio 0.5

# 75% compression
python experiments/baseline.py --num_examples 100 --compression_type fixed --compression_ratio 0.75
```

**Analysis Questions:**
1. What compression ratio gives best F1 vs memory tradeoff?
2. What types of questions fail with compression?
3. Does compressing Agent 1→2 hurt more than Agent 2→3?

### Step 5: Role-Specific Experiments (Week 4-6)

Run role-specific compression:

```bash
python experiments/baseline.py --num_examples 500 --compression_type role_specific --compression_ratio 0.5
```

Or run full comparison:

```bash
python experiments/baseline.py --num_examples 500 --comparison
```

**Success Criteria:**
- Role-specific F1 ≥ 90% of no-compression baseline
- Context reduction ≥ 40%
- Statistical significance: p < 0.05 vs fixed compression

### Step 6: Analysis and Visualization (Week 7-9)

Generate visualizations and analysis:

```bash
# Create plots
python experiments/analyze_results.py --results_dir results --plot

# Generate report
python experiments/analyze_results.py --results_dir results --report

# Statistical comparison
python experiments/analyze_results.py --compare "fixed" "role_specific"
```

## 📊 Expected Results

### Performance Targets

| Metric | Target | Method to Achieve |
|--------|--------|-------------------|
| Baseline F1 | >0.70 | No compression |
| Role-specific F1 | >0.68 | Our method with 50% compression |
| Fixed compression F1 | >0.63 | Fixed 50% compression |
| Context reduction | >40% | Both compression methods |
| Memory usage | <6GB | Careful model selection, batch=1 |
| Statistical significance | p<0.05 | Paired t-test vs fixed |

### Key Findings to Document

1. **When Role-Specific Helps:**
   - Multi-hop questions
   - Questions with specific entities/dates
   - Longer contexts

2. **Failure Modes:**
   - Very short contexts (nothing to compress)
   - Answer in first sentence (fixed works fine)
   - Model errors (unrelated to compression)

3. **Compression Statistics:**
   - Retriever output: ~800-1200 tokens
   - After compression: ~400-600 tokens
   - Reasoner output: ~600-900 tokens
   - After compression: ~300-500 tokens

## 🔧 Advanced Usage

### Custom Compression Strategy

```python
from compression.role_specific import RoleSpecificScorer, Clarifier

# Create custom scorer
scorer = RoleSpecificScorer("retriever", "reasoner")

# Get importance scores
scores = scorer.score_tokens(
    context=text,
    metadata={'question': question}
)

# Use custom clarifier
clarifier = Clarifier("retriever", "reasoner")
compressed = clarifier.clarify(
    context=text,
    metadata={'question': question},
    target_compression=0.4,
    min_sentences=3
)
```

### Custom Experiment Configuration

```python
from experiments.baseline import ExperimentRunner

runner = ExperimentRunner(
    model_name="gpt2",  # or "distilgpt2", "gpt2-medium"
    output_dir="custom_results"
)

results = runner.run_experiment(
    num_examples=200,
    compression_type="role_specific",
    compression_ratio=0.6,
    experiment_name="custom_experiment"
)
```

### Batch Processing

```python
# Run multiple experiments
ratios = [0.3, 0.4, 0.5, 0.6, 0.7]

for ratio in ratios:
    runner.run_experiment(
        num_examples=100,
        compression_type="role_specific",
        compression_ratio=ratio,
        experiment_name=f"role_specific_{int(ratio*100)}"
    )
```

## 🐛 Debugging

### Common Issues

**1. Out of Memory:**
```python
# Solution 1: Use smaller model
chain = AgentChain(model_name="distilgpt2")

# Solution 2: Reduce max length
chain.retriever.max_length = 512
chain.reasoner.max_length = 512
chain.verifier.max_length = 512
```

**2. Slow Processing:**
```bash
# Process fewer examples
python experiments/baseline.py --num_examples 10

# Or use CPU (slower but stable)
export CUDA_VISIBLE_DEVICES=""
```

**3. Dataset Download Fails:**
```python
# Specify cache directory
loader = SQuADLoader(cache_dir="/path/to/cache")
```

### Logging and Monitoring

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Track memory
from utils.memory_tracker import MemoryTracker
tracker = MemoryTracker()
tracker.log_memory("checkpoint_1")
# ... your code ...
tracker.log_memory("checkpoint_2")
tracker.print_memory_summary()
```

## 📈 Next Steps (Weeks 10-16)

### Week 10-12: Extended Validation

**TODO:**
1. Add HotpotQA dataset support
2. Add DROP dataset support
3. Run cross-dataset experiments
4. Implement additional baselines (RAG, attention-based)

### Week 13-16: Paper Writing

**TODO:**
1. Write paper sections using analysis outputs
2. Create publication-quality figures
3. Compile comprehensive results tables
4. Write abstract and introduction
5. Submit to target conference

## 📚 Code Organization Best Practices

### Adding New Agent

```python
from agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(role="custom", **kwargs)

    def get_prompt(self, question, context, **kwargs):
        return f"Custom prompt: {question}\n{context}"

    def process(self, question, context, metadata=None):
        # Your implementation
        pass
```

### Adding New Compression Strategy

```python
from compression.naive_compression import SentenceScorer

class MyCompressor:
    def compress(self, text: str, ratio: float) -> str:
        # Your compression logic
        return compressed_text
```

### Adding New Metric

```python
from utils.metrics import MetricsTracker

# Extend MetricsTracker
tracker = MetricsTracker()
tracker.custom_metric = []  # Add your metric

# Or create new tracker
class CustomMetricsTracker(MetricsTracker):
    def __init__(self):
        super().__init__()
        self.custom_metrics = []

    def add_custom_metric(self, value):
        self.custom_metrics.append(value)
```

## 🎓 Research Outputs

### Files to Include in Paper

1. **Results:**
   - `results/comparison_*.json` - All experimental results
   - `results/f1_comparison.png` - Main results figure
   - `results/compression_tradeoff.png` - Tradeoff analysis
   - `results/analysis_report.txt` - Statistical analysis

2. **Code:**
   - Full repository on GitHub
   - Key implementation: `compression/role_specific.py`
   - Experiment scripts: `experiments/baseline.py`

3. **Documentation:**
   - README.md
   - Implementation Guide (this file)
   - Research Plan

### Citation Template

```bibtex
@article{gadekar2025coc,
  title={Chain of Clarifications: Role-Specific Context Compression for Multi-Agent LLM Systems},
  author={Gadekar, Harshay},
  journal={IEEE Conference},
  year={2025}
}
```

---

**Implementation Status**: ✅ **COMPLETE**

All core components for Weeks 1-9 of the research plan have been implemented and are ready for experimental validation.

Last Updated: November 2025
