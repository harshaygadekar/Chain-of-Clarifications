"""
Reasoner Agent - Second agent in the chain

The Reasoner's role is to:
1. Receive extracted information from the Retriever
2. Apply logical reasoning to the question
3. Generate a candidate answer with supporting reasoning
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class ReasonerAgent(BaseAgent):
    """
    Reasoner Agent - Applies logical reasoning to generate answers.

    This agent takes the extracted information from the Retriever
    and uses it to formulate an answer to the question.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_length: int = 1024
    ):
        """Initialize the Reasoner agent."""
        super().__init__(
            role="reasoner",
            model_name=model_name,
            device=device,
            max_length=max_length
        )

    def get_prompt(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> str:
        """
        Construct reasoner-specific prompt.

        The reasoner needs to use the extracted information to
        formulate a logical answer.

        Args:
            question: The question to answer
            context: Extracted relevant information from Retriever
            **kwargs: Additional parameters

        Returns:
            Formatted prompt for the reasoner
        """
        prompt = f"""You are a reasoning specialist. Your task is to analyze the given information and provide a clear, logical answer to the question.

Question: {question}

Relevant Information:
{context}

Task:
1. Carefully read the question and the relevant information provided
2. Apply logical reasoning to connect the information to the question
3. Formulate a clear, concise answer
4. Provide your reasoning chain that led to this answer
5. If the information is insufficient, state what's missing

Your Analysis and Answer:"""

        return prompt

    def process(
        self,
        question: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process the extracted information and generate an answer.

        Args:
            question: The question to answer
            context: Relevant information from Retriever
            metadata: Additional metadata

        Returns:
            Dictionary containing:
                - output: Reasoning and answer
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - role: Agent role
        """
        if metadata is None:
            metadata = {}

        # Create the prompt
        prompt = self.get_prompt(question, context)

        # Count input tokens
        input_tokens = self.count_tokens(prompt)

        # Generate response
        logger.info(f"Reasoner processing question: {question[:100]}...")
        output = self.generate_response(
            prompt,
            max_new_tokens=min(450, self.max_length // 2)
        )

        # Count output tokens
        output_tokens = self.count_tokens(output)

        result = {
            'output': output,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'role': self.role,
            'question': question,
            'memory_usage': self.get_memory_usage()
        }

        logger.info(f"Reasoner generated {output_tokens} tokens of reasoning")

        return result

    def extract_answer(self, reasoning_output: str) -> str:
        """
        Extract the final answer from the reasoning output.

        Args:
            reasoning_output: The full reasoning text

        Returns:
            Extracted answer string
        """
        # Simple heuristic: look for "Answer:" or similar patterns
        import re

        # Try to find explicit answer markers
        patterns = [
            r'Answer:\s*(.+?)(?:\n|$)',
            r'The answer is\s*(.+?)(?:\n|\.)',
            r'Therefore,?\s*(.+?)(?:\n|\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, reasoning_output, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return first sentence
        sentences = reasoning_output.split('.')
        return sentences[0].strip() if sentences else reasoning_output[:100]
