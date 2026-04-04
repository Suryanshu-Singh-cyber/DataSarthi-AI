import numpy as np

class SystemCompatibilityChecker:
    
    @staticmethod
    def estimate_memory_usage(rows, columns, bytes_per_cell=8):
        """Estimate memory usage of dataset"""
        estimated_mb = (rows * columns * bytes_per_cell) / (1024 * 1024)
        return estimated_mb
    
    @staticmethod
    def check_compatibility(ram_gb, cpu_cores, dataset_rows, dataset_cols):
        """
        Check if dataset is compatible with system
        
        Returns:
        - compatible: boolean
        - message: string
        - recommended_max_rows: int
        """
        estimated_memory_mb = SystemCompatibilityChecker.estimate_memory_usage(dataset_rows, dataset_cols)
        estimated_memory_gb = estimated_memory_mb / 1024
        
        # Rule-based compatibility check
        memory_factor = 0.7  # Use 70% of RAM for safety
        available_ram_gb = ram_gb * memory_factor
        
        # Memory check
        memory_compatible = estimated_memory_gb <= available_ram_gb
        
        # CPU check (simple heuristic)
        cpu_compatible = cpu_cores >= 2 or dataset_rows < 10000
        
        compatible = memory_compatible and cpu_compatible
        
        # Generate message
        if memory_compatible:
            memory_msg = f"✅ Memory OK: ~{estimated_memory_gb:.2f} GB / {ram_gb} GB RAM"
        else:
            memory_msg = f"❌ Memory Issue: ~{estimated_memory_gb:.2f} GB > {ram_gb} GB RAM"
        
        if cpu_compatible:
            cpu_msg = f"✅ CPU OK: {cpu_cores} cores available"
        else:
            cpu_msg = f"⚠️ CPU Limited: {cpu_cores} cores may be slow for this dataset"
        
        # Recommended max rows
        available_bytes = available_ram_gb * 1024 * 1024 * 1024
        bytes_per_row = dataset_cols * 8
        recommended_max_rows = int(available_bytes / bytes_per_row) if bytes_per_row > 0 else 100000
        
        return {
            'compatible': compatible,
            'estimated_memory_gb': round(estimated_memory_gb, 2),
            'estimated_memory_mb': round(estimated_memory_mb, 2),
            'available_ram_gb': round(available_ram_gb, 2),
            'memory_message': memory_msg,
            'cpu_message': cpu_msg,
            'recommended_max_rows': recommended_max_rows,
            'recommendation': f"Recommended max rows: {recommended_max_rows:,}"
        }
    
    @staticmethod
    def suggest_optimizations(dataset_rows, dataset_cols, ram_gb):
        """Suggest optimizations for large datasets"""
        suggestions = []
        
        estimated_mb = SystemCompatibilityChecker.estimate_memory_usage(dataset_rows, dataset_cols)
        
        if estimated_mb > 500:  # >500MB
            suggestions.append("💡 Consider using chunking or sampling")
        
        if dataset_cols > 50:
            suggestions.append("💡 Apply dimensionality reduction (PCA)")
        
        if dataset_rows > 100000:
            suggestions.append("💡 Use incremental learning or use smaller sample")
        
        if estimated_mb / 1024 > ram_gb * 0.5:
            suggestions.append("💡 Reduce precision (float64 → float32)")
        
        if dataset_rows > 50000:
            suggestions.append("💡 Consider using a subset of your data for initial experiments")
        
        return suggestions

# Backward compatibility function
def check_system(ram_gb, cpu_cores, dataset_size_mb):
    """Simple function interface for backward compatibility"""
    # Estimate rows and columns from size (rough approximation)
    estimated_rows = int(dataset_size_mb * 1024 * 1024 / (8 * 10))  # Assume 10 columns
    result = SystemCompatibilityChecker.check_compatibility(
        ram_gb, cpu_cores, estimated_rows, 10
    )
    
    # Adapt the output for simple interface
    if result['compatible']:
        status = "✅ Compatible"
        message = result['memory_message']
    else:
        status = "⚠️ May be too large"
        message = result['memory_message']
    
    return {'status': status, 'message': message}
