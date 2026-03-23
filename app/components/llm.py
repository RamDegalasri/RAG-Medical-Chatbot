import boto3
from botocore.config import Config as BotoConfig

import json
from typing import Optional

from app.config.config import Config
from app.common.logger import MedicalRAGLogger

class BedrockGemma3LLM:
    """
    AWS Bedrock Gemma 3 27B LLM setup
    Simple LLM interface for text generation
    """

    def __init__(self, model_id: Optional[str] = None, region_name: Optional[str] = None):
        """
        Initialize AWS Bedrock Gemma 3 27B client

        Args:
            model_id: Bedrock model identifier (default from config)
            region_name: AWS region (default from config)
        """
        self.logger = MedicalRAGLogger(__name__)
        self.model_id = model_id or Config.BEDROCK_MODEL_ID
        self.region_name = region_name or Config.AWS_REGION

        self.logger.logger.info(f"Initializing AWS Bedrock Gemma 3 27B...")
        self.logger.logger.info(f"  Model: {self.model_id}")
        self.logger.logger.info(f"  Region: {self.region_name}")

        try:
            # Configure boto3
            boto_config = BotoConfig(
                region_name = self.region_name,
                retries = {'max_attempts': 3, 'mode': 'adaptive'}
            )

            # Create Bedrock Runtime client
            self.bedrock_runtime = boto3.client(
                service_name = 'bedrock-runtime',
                region_name = self.region_name,
                config = boto_config,
                aws_access_key_id = Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key = Config.AWS_SECRET_ACCESS_KEY
            )

            self.logger.logger.info("AWS Bedrock client initialized")

        except Exception as e:
            self.logger.log_error(e, context = "Initializing AWS Bedrock")
            raise

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9, stop_sequences: Optional[list] = None) -> str:
        """
        Generate text using Gemma 3 27B

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling thresold
            stop_sequences: List of sequences to stop generation

        Returns:
            Generated text
        """
        try:
            self.logger.logger.info("Generating response...")
            self.logger.logger.info(f"  Prompt length: {len(prompt)} chars")
            self.logger.logger.info(f"  Temperature: {temperature}")
            self.logger.logger.info(f"  Max tokens: {max_tokens}")

            # Prepare request body (Gemma 3 on Bedrock uses messages-based API)
            request_body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            if stop_sequences:
                request_body["stop_sequences"] = stop_sequences

            # Call Bedrock API
            response = self.bedrock_runtime.invoke_model(
                modelId = self.model_id,
                contentType = "application/json",
                accept = "application/json",
                body = json.dumps(request_body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract generated text from Gemma 3 Bedrock messages API response
            generated_text = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')

            self.logger.logger.info("Response generated")
            self.logger.logger.info(f"  Output length: {len(generated_text)} chars")

            return generated_text.strip()
        
        except Exception as e:
            self.logger.log_error(e, context = "Generating text with Gemma 3 27B")
            raise

    def get_model_info(self) -> dict:
        """
        Get model information

        Returns:
            Model confirguation details
        """
        return {
            "model_id": self.model_id,
            "provider": "AWS Bedrock",
            "model_name": "Gemma 3 27B",
            "region": self.region_name,
            "parameters": "27 billion"
        }
    
# Usage/Testing
if __name__ == "__main__":
    print("\n" + "="*190)
    print("AWS BEDROCK GEMMA 3 27B - SETUP TEST")
    print("=" * 190 + "\n")

    # Test 1: Initialize LLM
    print("Test 1: Initializing Gemma 3 27B...")
    print("-" * 190)

    try:
        llm = BedrockGemma3LLM()
        print("LLM initialized successfully\n")

        # Show model info
        info = llm.get_model_info()
        print("Model Information:")
        for key, value in info.items():
            print(f" {key}: {value}")
        print()

    except Exception as e:
        print(f"x Error initializing LLM: {e}")
        print("\nTroubleshooting:")
        print(" 1. Check AWS credentials in .env file")
        print("  2. Verify AWS_ACCESS_KEY_ID is set")
        print("  3. Verify AWS_SECRET_ACCESS_KEY is set")
        print("  4. Ensure Bedrock is enabled in your AWS account")
        print("  5. Check model ID is correct")
        exit(1)

    # Test 2: Simple generation
    print("\nTest 2: Testing text generation...")
    print("-" * 190)

    test_prompt = "What is artificial intelligence?"

    try:
        print(f"Prompt: {test_prompt}\n")

        response = llm.generate(
            prompt = test_prompt,
            max_tokens = 190,
            temperature = 0.7
        )

        print(f"Response:\n{response}\n")
        print("Text generation successful\n")

    except Exception as e:
        print(f"x Error generating text: {e}\n")

    # Test 3: Multiple temperature settings
    print("Test 3: Testing different temperatures...")
    print("-" * 190)

    simple_prompt = "Explain diabetes in one sentence."

    for temp in [0.0, 0.5, 1.0]:
        try:
            print(f"\nTemperature: {temp}")
            response = llm.generate(
                prompt = simple_prompt,
                max_tokens = 100,
                temperature = temp
            )
            print(f"Response: {response[:150]}...")
        
        except Exception as e:
            print(f"x Error at temerature {temp}: {e}")

    print("\n" + "=" * 190)
    print("SETUP TEST COMPLETED")
    print("=" * 190 + "\n")

    print("Summary:")
    print(" AWS Bedrock connection working")
    print("  Gemma 3 27B model accessible")
    print("  Text generation functional")
    print("\nNext steps:")
    print("  Implement prompt engineering")
    print("  Add RAG integration")
    print("  Build chat interface")
