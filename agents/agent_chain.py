"""
Agent Chain Orchestrator

Coordinates the multi-agent pipeline: Retriever → Reasoner → Verifier
Handles context passing, compression, and result aggregation.
"""

from typing import Dict, Optional, Any
from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.verifier import VerifierAgent
from compression.naive_compression import NaiveCompressor
from compression.role_specific import Clarifier
import time
import logging

logger = logging.getLogger(__name__)


class AgentChain:
    """
    Orchestrates the three-agent chain with optional compression.

    Supports multiple compression strategies:
    - No compression
    - Fixed-ratio compression
    - Role-specific adaptive compression
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        compression_type: str = "none",
        compression_ratio: float = 0.5
    ):
        """
        Initialize the agent chain.

        Args:
            model_name: Model to use for all agents
            device: Computing device
            compression_type: Type of compression (none, fixed, role_specific)
            compression_ratio: Compression ratio if applicable
        """
        self.model_name = model_name
        self.device = device
        self.compression_type = compression_type
        self.compression_ratio = compression_ratio

        # Initialize agents
        logger.info("Initializing agent chain...")
        self.retriever = RetrieverAgent(
            model_name=model_name,
            device=device
        )
        self.reasoner = ReasonerAgent(
            model_name=model_name,
            device=device
        )
        self.verifier = VerifierAgent(
            model_name=model_name,
            device=device
        )

        # Initialize compression modules
        self._init_compression()

        logger.info(
            f"Agent chain initialized with {compression_type} compression"
        )

    def _init_compression(self):
        """Initialize compression modules based on type."""
        if self.compression_type == "fixed":
            self.compressor_1_2 = NaiveCompressor(
                compression_ratio=self.compression_ratio,
                strategy="first_n"
            )
            self.compressor_2_3 = NaiveCompressor(
                compression_ratio=self.compression_ratio,
                strategy="first_n"
            )
        elif self.compression_type == "role_specific":
            self.clarifier_1_2 = Clarifier("retriever", "reasoner")
            self.clarifier_2_3 = Clarifier("reasoner", "verifier")
        else:
            # No compression
            self.compressor_1_2 = None
            self.compressor_2_3 = None

    def process(
        self,
        question: str,
        document: str,
        track_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Process a question through the agent chain.

        Args:
            question: Question to answer
            document: Source document/passage
            track_metrics: Whether to track detailed metrics

        Returns:
            Dictionary with final answer and metrics
        """
        start_time = time.time()
        results = {
            'question': question,
            'success': False,
            'error': None
        }

        try:
            # Agent 1: Retriever
            logger.info("=== Agent 1: Retriever ===")
            retriever_result = self.retriever.process(question, document)

            context_1 = retriever_result['output']
            logger.info(f"Retriever output: {len(context_1)} chars")

            # Compress for Agent 2
            if self.compression_type == "fixed":
                context_1_compressed = self.compressor_1_2.compress(context_1)
            elif self.compression_type == "role_specific":
                context_1_compressed = self.clarifier_1_2.clarify(
                    context_1,
                    metadata={'question': question},
                    target_compression=self.compression_ratio
                )
            else:
                context_1_compressed = context_1

            logger.info(
                f"Context 1→2: {len(context_1)} → {len(context_1_compressed)} chars"
            )

            # Agent 2: Reasoner
            logger.info("=== Agent 2: Reasoner ===")
            reasoner_result = self.reasoner.process(
                question,
                context_1_compressed
            )

            context_2 = reasoner_result['output']
            logger.info(f"Reasoner output: {len(context_2)} chars")

            # Compress for Agent 3
            if self.compression_type == "fixed":
                context_2_compressed = self.compressor_2_3.compress(context_2)
            elif self.compression_type == "role_specific":
                context_2_compressed = self.clarifier_2_3.clarify(
                    context_2,
                    metadata={
                        'question': question,
                        'retrieval': context_1_compressed
                    },
                    target_compression=self.compression_ratio
                )
            else:
                context_2_compressed = context_2

            logger.info(
                f"Context 2→3: {len(context_2)} → {len(context_2_compressed)} chars"
            )

            # Agent 3: Verifier
            logger.info("=== Agent 3: Verifier ===")
            verifier_result = self.verifier.process(
                question,
                context_2_compressed
            )

            final_answer = verifier_result['final_answer']
            logger.info(f"Final answer: {final_answer[:100]}...")

            # Aggregate results
            end_time = time.time()
            results.update({
                'success': True,
                'final_answer': final_answer,
                'latency': end_time - start_time,
                'retriever_output': context_1,
                'reasoner_output': context_2,
                'verifier_output': verifier_result['output'],
                'context_sizes': {
                    'retriever': len(context_1.split()),
                    'retriever_compressed': len(context_1_compressed.split()),
                    'reasoner': len(context_2.split()),
                    'reasoner_compressed': len(context_2_compressed.split()),
                },
                'token_counts': {
                    'retriever_input': retriever_result['input_tokens'],
                    'retriever_output': retriever_result['output_tokens'],
                    'reasoner_input': reasoner_result['input_tokens'],
                    'reasoner_output': reasoner_result['output_tokens'],
                    'verifier_input': verifier_result['input_tokens'],
                    'verifier_output': verifier_result['output_tokens'],
                }
            })

            # Add memory info
            if track_metrics:
                results['memory_usage'] = self.verifier.get_memory_usage()

        except Exception as e:
            logger.error(f"Error in agent chain: {e}", exc_info=True)
            results['success'] = False
            results['error'] = str(e)
            results['latency'] = time.time() - start_time

        return results

    def cleanup(self):
        """Clean up all agents and free memory."""
        logger.info("Cleaning up agent chain...")
        self.retriever.cleanup()
        self.reasoner.cleanup()
        self.verifier.cleanup()

    def __repr__(self) -> str:
        return (
            f"AgentChain(model={self.model_name}, "
            f"compression={self.compression_type})"
        )
