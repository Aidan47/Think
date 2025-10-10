import csv
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

class TaskLogger:
    """
    A class that handles logging arbitrary information to CSV files
    while providing a progress tracking display.
    """
    
    def __init__(self, csv_file_path: str, fieldnames: Optional[List[str]] = None):
        """
        Initialize the TaskLogger.
        
        Args:
            csv_file_path: Path to the CSV file for logging
            fieldnames: List of column names for the CSV. If None, will be set from first log entry.
        """
        self.csv_file_path = csv_file_path
        self.fieldnames = fieldnames
        self.completed_tasks = 0
        self.total_tasks = 0
        self._csv_file = None
        self._csv_writer = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)
        
    def __enter__(self):
        """Context manager entry."""
        self.start_logging()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_logging()
        
    def start_logging(self):
        """Initialize the CSV file and writer."""
        self._csv_file = open(self.csv_file_path, 'a', newline='', encoding='utf-8')
        
        # Write header if file is empty or new
        file_exists = os.path.exists(self.csv_file_path) and os.path.getsize(self.csv_file_path) > 0
        
        if not file_exists and self.fieldnames:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
            self._csv_writer.writeheader()
        else:
            # If fieldnames not provided, we'll set them from the first write
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames or [])
    
    def stop_logging(self):
        """Close the CSV file."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
    
    def set_total_tasks(self, total: int):
        """Set the total number of tasks for progress tracking."""
        self.total_tasks = total
        self.completed_tasks = 0
        self._update_display()
    
    def log_task(self, data: Dict[str, Any], update_progress: bool = True):
        """
        Log task data to CSV and optionally update progress.
        
        Args:
            data: Dictionary of data to log (keys should match fieldnames)
            update_progress: Whether to increment completed tasks counter
        """
        if self._csv_writer is None:
            self.start_logging()
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Set fieldnames from first data if not already set
        if self.fieldnames is None:
            self.fieldnames = list(data.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
            
            # Write header if file is empty
            if os.path.getsize(self.csv_file_path) == 0:
                self._csv_writer.writeheader()
        
        self._csv_writer.writerow(data)
        self._csv_file.flush()  # Ensure data is written immediately
        
        if update_progress:
            self.completed_tasks += 1
            self._update_display()
    
    def increment_progress(self, count: int = 1):
        """Increment the progress counter without logging data."""
        self.completed_tasks += count
        self._update_display()
    
    def _update_display(self):
        """Update the progress display."""
        if self.total_tasks > 0:
            percentage = (self.completed_tasks / self.total_tasks) * 100
            print(f"\rCompleted: {self.completed_tasks}/{self.total_tasks} ({percentage:.1f}%)", 
                  end="", flush=True)
        else:
            print(f"\rCompleted: {self.completed_tasks} tasks", end="", flush=True)
    
    def finish(self):
        """Complete the logging session with a final display update."""
        self._update_display()
        print()  # Move to next line
        self.stop_logging()

class ProgressBar:
    """
    A simple progress bar for terminal display.
    """
    
    def __init__(self, total: int, width: int = 50, description: str = "Progress"):
        """
        Initialize the progress bar.
        
        Args:
            total: Total number of tasks
            width: Width of the progress bar in characters
            description: Description to display before the bar
        """
        self.total = total
        self.width = width
        self.description = description
        self.current = 0
        
    def update(self, n: int = 1):
        """Update the progress by n steps."""
        self.current += n
        self.current = min(self.current, self.total)
        self._display()
    
    def _display(self):
        """Display the progress bar."""
        percentage = self.current / self.total
        filled_length = int(self.width * percentage)
        bar = "█" * filled_length + "░" * (self.width - filled_length)
        print(f"\r{self.description}: |{bar}| {self.current}/{self.total} ({percentage:.1%})", 
              end="", flush=True)
    
    def finish(self):
        """Complete the progress bar display."""
        self._display()
        print()  # Move to next line

# Example usage and demonstration
def demonstrate_logger():
    """Demonstrate the usage of TaskLogger with progress tracking."""
    
    # Example 1: Using TaskLogger with automatic progress tracking
    print("Example 1: TaskLogger with progress tracking")
    print("-" * 50)
    
    fieldnames = ['timestamp', 'task_id', 'status', 'result', 'processing_time']
    
    with TaskLogger('task_log.csv', fieldnames) as logger:
        logger.set_total_tasks(20)
        
        for i in range(20):
            # Simulate task processing
            time.sleep(0.5)
            
            # Log task data
            task_data = {
                'task_id': i,
                'status': 'completed',
                'result': f"result_{i}",
                'processing_time': 0.5
            }
            logger.log_task(task_data)
    
    print("\n")  # Add spacing
    
    # Example 2: Using ProgressBar separately
    print("Example 2: Standalone ProgressBar")
    print("-" * 50)
    
    progress = ProgressBar(total=15, description="Downloading")
    
    for i in range(15):
        time.sleep(1)
        progress.update()
    
    progress.finish()
    
    # Example 3: Manual progress tracking with TaskLogger
    print("\nExample 3: Manual progress updates")
    print("-" * 50)
    
    logger = TaskLogger('manual_log.csv', ['timestamp', 'operation', 'details'])
    logger.start_logging()
    logger.set_total_tasks(10)
    
    for i in range(10):
        time.sleep(0.5)
        
        # Log without automatic progress update
        logger.log_task({
            'operation': f'step_{i}',
            'details': f'Processing item {i}'
        }, update_progress=False)
        
        # Manual progress update
        logger.increment_progress()
    
    logger.finish()

class TestReporter:
    def __init__(self, total_tests: int):
        self.total_tests = total_tests
        self.completed = 0
        self.correct = 0
    
    def update(self, is_correct: bool):
        self.completed += 1
        if is_correct:
            self.correct += 1
        self._display()
    
    def _display(self):
        accuracy = (self.correct / self.completed) * 100 if self.completed > 0 else 0
        print(f"\r✅ {self.completed}/{self.total_tests} | "
              f"Accuracy: {accuracy:.1f}% ({self.correct}/{self.completed})", 
              end="", flush=True)
    
    def final_report(self):
        accuracy = (self.correct / self.total_tests) * 100
        print(f"\n🎯 Final: {accuracy:.1f}% accuracy "
              f"({self.correct}/{self.total_tests} correct)")
        return accuracy

def batch_processing_example():
    """Example of batch processing with detailed logging."""
    
    print("\nBatch Processing Example")
    print("-" * 50)
    
    fieldnames = [
        'timestamp', 'file_name', 'file_size', 'operation', 
        'status', 'error_message', 'processing_time'
    ]
    
    # Simulated file processing
    files_to_process = [
        {'name': 'document1.pdf', 'size': 1024},
        {'name': 'image2.jpg', 'size': 2048},
        {'name': 'data3.csv', 'size': 512},
        {'name': 'report4.docx', 'size': 3072},
    ]
    
    with TaskLogger('file_processing_log.csv', fieldnames) as logger:
        logger.set_total_tasks(len(files_to_process))
        
        for file_info in files_to_process:
            start_time = time.time()
            
            try:
                # Simulate processing
                time.sleep(1)
                
                # Simulate occasional error
                if file_info['name'] == 'image2.jpg':
                    raise ValueError("Invalid image format")
                
                # Log successful processing
                logger.log_task({
                    'file_name': file_info['name'],
                    'file_size': file_info['size'],
                    'operation': 'compress',
                    'status': 'success',
                    'processing_time': time.time() - start_time
                })
                
            except Exception as e:
                # Log error
                logger.log_task({
                    'file_name': file_info['name'],
                    'file_size': file_info['size'],
                    'operation': 'compress',
                    'status': 'error',
                    'error_message': str(e),
                    'processing_time': time.time() - start_time
                })

if __name__ == "__main__":
    demonstrate_logger()
    batch_processing_example()
    
    # Simple usage:
    reporter = TestReporter(total_tests=244)
    for test_case in range(244):
        time.sleep(0.25)
        result = random.randint(0,1)
        reporter.update(is_correct=bool(result))
    final_accuracy = reporter.final_report()
    
    print("\nAll examples completed! Check the generated CSV files.")