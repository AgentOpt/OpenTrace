"""
Example demonstrating async operations and concurrent optimization in Trace.

This example shows how to:
- Use async bundle functions for non-blocking operations
- Run concurrent optimizations with asyncio
- Handle async LLM calls efficiently
- Implement async data loading and processing
- Coordinate multiple async trace operations
"""

import asyncio
import time
import random
from typing import List, Any
from opto.trace import node, bundle, GRAPH
from opto.trace.nodes import ParameterNode
from opto.optimizers import OptoPrime
from opto.utils.llm import AutoGenLLM


# Example 1: Async Bundle Functions
@bundle()
async def async_api_call(query):
    """Simulate an async API call with tracing."""
    # Simulate network delay
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # Simulate API response
    response = f"API response for: {query}"
    return response


@bundle()
async def async_data_processor(data):
    """Process data asynchronously with tracing."""
    # Simulate CPU-bound processing
    await asyncio.sleep(0.2)
    
    if isinstance(data, list):
        return [f"Processed: {item}" for item in data]
    return f"Processed: {data}"


# Example 2: Async Optimization Loop
class AsyncOptimizer:
    """Demonstrates async optimization patterns."""
    
    def __init__(self):
        self.parameters = []
        self.optimizer = None
    
    async def initialize_parameters(self, n_params: int):
        """Initialize parameters asynchronously."""
        tasks = []
        for i in range(n_params):
            # Simulate async parameter initialization (e.g., from database)
            async def create_param(idx):
                await asyncio.sleep(0.1)
                return ParameterNode(
                    f"initial_value_{idx}",
                    name=f"param_{idx}",
                    description=f"Parameter {idx} to optimize"
                )
            tasks.append(create_param(i))
        
        self.parameters = await asyncio.gather(*tasks)
        print(f"Initialized {len(self.parameters)} parameters asynchronously")
        return self.parameters
    
    async def async_forward(self, params):
        """Async forward pass with multiple parameters."""
        # Run async operations concurrently
        tasks = []
        for p in params:
            tasks.append(async_api_call(p))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def optimize_step(self):
        """Single async optimization step."""
        # Forward pass
        results = await self.async_forward(self.parameters)
        
        # Simulate feedback computation
        await asyncio.sleep(0.1)
        feedback = f"Aggregated feedback from {len(results)} results"
        
        # Simulate parameter update
        for i, param in enumerate(self.parameters):
            param._data = f"updated_value_{i}_{time.time():.2f}"
        
        return feedback


# Example 3: Concurrent Trace Operations
class ConcurrentTracer:
    """Demonstrates concurrent tracing patterns."""
    
    @bundle()
    async def fetch_data(self, source: str):
        """Fetch data from a source asynchronously."""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return f"Data from {source}"
    
    @bundle()
    async def process_batch(self, batch: List[Any]):
        """Process a batch of items concurrently."""
        tasks = []
        for item in batch:
            async def process_item(x):
                await asyncio.sleep(0.1)
                return f"Processed: {x}"
            tasks.append(process_item(item))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def pipeline(self, sources: List[str]):
        """Async pipeline with concurrent stages."""
        # Stage 1: Fetch data concurrently
        fetch_tasks = [self.fetch_data(node(src)) for src in sources]
        raw_data = await asyncio.gather(*fetch_tasks)
        
        # Stage 2: Process in batches
        batch_size = 2
        all_results = []
        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i+batch_size]
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
        
        return all_results


# Example 4: Async Context Manager for Tracing
class AsyncTraceContext:
    """Context manager for async trace operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.operations = []
    
    async def __aenter__(self):
        """Enter async context."""
        self.start_time = time.time()
        print(f"Starting async trace context: {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        elapsed = time.time() - self.start_time
        print(f"Completed {self.name} in {elapsed:.2f}s")
        print(f"  Executed {len(self.operations)} operations")
    
    @bundle()
    async def traced_operation(self, op_name: str, data: Any):
        """Execute and trace an async operation."""
        self.operations.append(op_name)
        await asyncio.sleep(0.1)
        return f"Result of {op_name} on {data}"


# Example 5: Async Minibatch Optimization
class AsyncMinibatchOptimizer:
    """Demonstrates async minibatch optimization."""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.model_param = ParameterNode(
            "initial_model",
            name="model",
            description="Model parameters"
        )
    
    @bundle()
    async def process_sample(self, sample, model):
        """Process a single sample asynchronously."""
        await asyncio.sleep(random.uniform(0.05, 0.15))
        # Simulate loss computation
        loss = random.random()
        return loss
    
    async def process_minibatch(self, batch):
        """Process a minibatch concurrently."""
        tasks = []
        for sample in batch:
            tasks.append(self.process_sample(
                node(sample),
                self.model_param
            ))
        
        losses = await asyncio.gather(*tasks)
        avg_loss = sum(losses) / len(losses)
        return avg_loss
    
    async def train_epoch(self, dataset):
        """Train one epoch with async minibatch processing."""
        total_loss = 0.0
        n_batches = 0
        
        # Process dataset in minibatches
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i+self.batch_size]
            batch_loss = await self.process_minibatch(batch)
            total_loss += batch_loss
            n_batches += 1
            
            # Simulate parameter update
            self.model_param._data = f"model_epoch_{n_batches}"
        
        return total_loss / n_batches


# Example 6: Async LLM Optimization
class AsyncLLMOptimizer:
    """Demonstrates async LLM-based optimization."""
    
    def __init__(self):
        self.prompts = []
        for i in range(3):
            self.prompts.append(ParameterNode(
                f"Initial prompt {i}",
                name=f"prompt_{i}",
                description=f"Prompt variant {i}"
            ))
    
    @bundle()
    async def async_llm_call(self, prompt):
        """Simulate async LLM API call."""
        # Simulate LLM latency
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Simulate LLM response
        score = random.random()
        return f"Response to '{prompt}' with score {score:.2f}"
    
    async def evaluate_prompts_concurrently(self):
        """Evaluate all prompts concurrently."""
        tasks = []
        for prompt in self.prompts:
            tasks.append(self.async_llm_call(prompt))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def optimize_prompts(self, n_iterations: int = 3):
        """Optimize prompts with concurrent evaluation."""
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration + 1}:")
            
            # Evaluate all prompts concurrently
            start = time.time()
            results = await self.evaluate_prompts_concurrently()
            elapsed = time.time() - start
            
            print(f"  Evaluated {len(self.prompts)} prompts in {elapsed:.2f}s (concurrent)")
            
            # Update prompts based on results
            for i, (prompt, result) in enumerate(zip(self.prompts, results)):
                if "score 0." in result.data and float(result.data.split()[-1]) < 0.5:
                    # Low score, update prompt
                    prompt._data = f"Improved prompt {i} (iter {iteration})"
            
            # Compare with sequential timing
            sequential_time = len(self.prompts) * 1.0  # Average 1s per LLM call
            print(f"  Sequential would take ~{sequential_time:.2f}s")
            print(f"  Speedup: {sequential_time/elapsed:.1f}x")


async def main():
    """Main async function demonstrating various patterns."""
    
    print("=" * 60)
    print("Async Trace Operations Example")
    print("=" * 60)
    
    # Example 1: Basic async bundle usage
    print("\n1. Basic Async Bundle Functions")
    print("-" * 40)
    
    query = node("What is async programming?")
    response = await async_api_call(query)
    print(f"Query: {query.data}")
    print(f"Response: {response.data}")
    
    # Example 2: Concurrent async operations
    print("\n2. Concurrent Async Operations")
    print("-" * 40)
    
    queries = [node(f"Query {i}") for i in range(5)]
    start = time.time()
    
    # Run concurrently
    tasks = [async_api_call(q) for q in queries]
    responses = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    
    print(f"Processed {len(queries)} queries in {concurrent_time:.2f}s (concurrent)")
    print(f"Sequential would take ~{len(queries) * 0.3:.2f}s")
    
    # Example 3: Async optimization
    print("\n3. Async Optimization Loop")
    print("-" * 40)
    
    optimizer = AsyncOptimizer()
    await optimizer.initialize_parameters(3)
    
    for step in range(2):
        print(f"\nOptimization step {step + 1}:")
        feedback = await optimizer.optimize_step()
        print(f"  Feedback: {feedback}")
    
    # Example 4: Concurrent pipeline
    print("\n4. Concurrent Processing Pipeline")
    print("-" * 40)
    
    tracer = ConcurrentTracer()
    sources = ["database", "api", "cache", "file"]
    results = await tracer.pipeline(sources)
    print(f"Pipeline processed {len(sources)} sources")
    for r in results:
        print(f"  - {r.data}")
    
    # Example 5: Async context manager
    print("\n5. Async Context Manager")
    print("-" * 40)
    
    async with AsyncTraceContext("data_processing") as ctx:
        result1 = await ctx.traced_operation("step1", node("data1"))
        result2 = await ctx.traced_operation("step2", result1)
        result3 = await ctx.traced_operation("step3", result2)
        print(f"Final result: {result3.data}")
    
    # Example 6: Async minibatch training
    print("\n6. Async Minibatch Training")
    print("-" * 40)
    
    trainer = AsyncMinibatchOptimizer(batch_size=4)
    dataset = [f"sample_{i}" for i in range(12)]
    
    start = time.time()
    avg_loss = await trainer.train_epoch(dataset)
    elapsed = time.time() - start
    
    print(f"Trained on {len(dataset)} samples in {elapsed:.2f}s")
    print(f"Average loss: {avg_loss.data:.4f}" if hasattr(avg_loss, 'data') else f"Average loss: {avg_loss:.4f}")
    
    # Example 7: Async LLM optimization
    print("\n7. Async LLM Optimization")
    print("-" * 40)
    
    llm_opt = AsyncLLMOptimizer()
    await llm_opt.optimize_prompts(n_iterations=2)
    
    print("\n" + "=" * 60)
    print("Async operations enable efficient concurrent optimization!")
    print("Key benefits:")
    print("  - Non-blocking I/O operations")
    print("  - Concurrent parameter evaluation")
    print("  - Efficient LLM API usage")
    print("  - Scalable minibatch processing")
    print("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())