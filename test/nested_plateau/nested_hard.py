
def nested_hard(timestamp: int):
    # --- Configuration ---
    # Outer Cycle: 1000 blocks (1 Block = 100000 sec)
    # Inner Split: 1000 slots per block
    # Target: The last slot of the last block in a cycle
    
    # 1. Outer Plateau (The "Week" check)
    # Block Index = timestamp // 100000
    # Cycle Position = Block Index % 1000
    # Target: 999 (Last day of the cycle)
    
    block_idx = timestamp // 100000
    cycle_pos = block_idx % 1000
    
    # Gradient Killing: Convert to string to hide distance
    if cycle_pos == 999:
        
        # 2. Inner Plateau (The "Hour" check)
        # Offset inside the block = timestamp % 100000
        # Slot Size = 100000 // 1000
        # Slot Index = Offset // Slot Size
        # Target: 500 (Middle slot)
        
        offset = timestamp % 100000
        slot_size = 100000 // 1000
        slot_idx = offset // slot_size
        
        if slot_idx == 500:
            return 0.0  # Success!
        
        # Inner Plateau Penalty (Wrong Hour)
        return 50.0
            
    # Outer Plateau Penalty (Wrong Day)
    return 100.0
