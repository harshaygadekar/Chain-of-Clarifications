"""
Base Agent Class for Chain of Clarifications System

This module provides the base class for all agents in the multi-agent chain.
Each agent performs a specific role: Retrieval, Reasoning, or Verification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents in the chain.

    Attributes:
        role (str): The role of this agent (retriever, reasoner, verifier)
        model_name (str): Name of the language model to use
        device (str): Device to run the model on (cuda/cpu)
        max_length (int): Maximum sequence length for generation
    """

    def __init__(
        self,
        role: str,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize the base agent.

        Args:
            role: Role identifier for this agent
            model_name: Hugging Face model identifier
            device: Computing device (auto-detected if None)
            max_length: Maximum tokens for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.role = role
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing {self.role} agent on {self.device}")

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model loaded successfully for {self.role}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate a response using the language model.

        Args:
            prompt: Input prompt for the model
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text: Input text to count

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def process(
        self,
        question: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input and generate output.
        This method should be overridden by child classes.

        Args:
            question: The question to answer
            context: Context information from previous agents
            metadata: Additional metadata

        Returns:
            Dictionary containing output and metadata
        """
        raise NotImplementedError("Subclasses must implement process()")

    def get_prompt(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> str:
        """
        Construct the prompt for this agent.
        Should be overridden by child classes.

        Args:
            question: The question
            context: Context information
            **kwargs: Additional parameters

        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement get_prompt()")

    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"{self.role} agent cleaned up")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory statistics
        """
        memory_info = {}

        if torch.cuda.is_available():
            memory_info['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_info['reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_info['max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory_info['allocated_mb'] = 0
            memory_info['reserved_mb'] = 0
            memory_info['max_allocated_mb'] = 0

        return memory_info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.role}, model={self.model_name}, device={self.device})"
