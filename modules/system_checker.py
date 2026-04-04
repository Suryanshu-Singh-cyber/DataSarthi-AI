def check_system(rows, cols, ram_gb):
    size_bytes = rows * cols * 8
    size_mb = size_bytes / (1024**2)

    if size_mb > ram_gb * 1024 * 0.5:
        return "⚠️ Dataset too large for your system"
    else:
        return "✅ Dataset can run smoothly"
