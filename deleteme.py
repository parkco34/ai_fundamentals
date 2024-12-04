#!/usr/bin/env python
from queue import Queue, PriorityQueue
import time
import random
from typing import List, Tuple
import statistics

class QueuePerformanceTester:
    def __init__(self, num_operations: int, num_trials: int):
        self.num_operations = num_operations
        self.num_trials = num_trials

    def generate_test_data(self) -> List[Tuple[str, int]]:
        """Generate a sequence of enqueue/dequeue operations with random priorities"""
        operations = []
        # Ensure we have a balanced number of enqueues and dequeues
        items_in_queue = 0

        for _ in range(self.num_operations):
            if items_in_queue == 0 or random.random() < 0.5:
                # Enqueue operation
                priority = random.randint(1, 100)
                operations.append(("enqueue", priority))
                items_in_queue += 1
            else:
                # Dequeue operation
                operations.append(("dequeue", None))
                items_in_queue -= 1

        # Add dequeue operations to empty the queue
        while items_in_queue > 0:
            operations.append(("dequeue", None))
            items_in_queue -= 1

        return operations

    def test_fifo_queue(self, operations: List[Tuple[str, int]]) -> float:
        """Test FIFO queue performance"""
        queue = Queue()
        start_time = time.perf_counter()

        for operation, value in operations:
            if operation == "enqueue":
                queue.put(value)
            else:
                queue.get()

        end_time = time.perf_counter()
        return end_time - start_time

    def test_priority_queue(self, operations: List[Tuple[str, int]]) -> float:
        """Test Priority Queue performance"""
        queue = PriorityQueue()
        start_time = time.perf_counter()

        for operation, value in operations:
            if operation == "enqueue":
                queue.put(value)
            else:
                queue.get()

        end_time = time.perf_counter()
        return end_time - start_time

    def run_comparison(self) -> dict:
        """Run multiple trials and collect statistics"""
        fifo_times = []
        priority_times = []

        for trial in range(self.num_trials):
            operations = self.generate_test_data()

            fifo_time = self.test_fifo_queue(operations)
            priority_time = self.test_priority_queue(operations)

            fifo_times.append(fifo_time)
            priority_times.append(priority_time)

        results = {
            'fifo': {
                'mean': statistics.mean(fifo_times),
                'std_dev': statistics.stdev(fifo_times),
                'min': min(fifo_times),
                'max': max(fifo_times)
            },
            'priority': {
                'mean': statistics.mean(priority_times),
                'std_dev': statistics.stdev(priority_times),
                'min': min(priority_times),
                'max': max(priority_times)
            }
        }

        return results

# Run the experiment
def main():
    # Test with different queue sizes
    sizes = [1000, 10000, 100000]
    trials = 10

    for size in sizes:
        print(f"\nTesting with {size} operations:")
        tester = QueuePerformanceTester(size, trials)
        results = tester.run_comparison()

        print(f"FIFO Queue:")
        print(f"  Mean time: {results['fifo']['mean']:.6f} seconds")
        print(f"  Std Dev:   {results['fifo']['std_dev']:.6f} seconds")
        print(f"Priority Queue:")
        print(f"  Mean time: {results['priority']['mean']:.6f} seconds")
        print(f"  Std Dev:   {results['priority']['std_dev']:.6f} seconds")
        print(f"Ratio (Priority/FIFO): {results['priority']['mean']/results['fifo']['mean']:.2f}x")

if __name__ == "__main__":
    main()
