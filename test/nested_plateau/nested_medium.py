
def nested_medium(timestamp: int):
    # --- Configuration ---
    # Outer Cycle: 100 blocks (1 Block = 100000 sec)
    # Inner Split: 100 slots per block
    # Target: The last slot of the last block in a cycle
    
    # 1. Outer Plateau (The "Week" check)
    # Block Index = timestamp // 100000
    # Cycle Position = Block Index % 100
    # Target: 99 (Last day of the cycle)
    
    block_idx = timestamp // 100000
    cycle_pos = block_idx % 100
    
    # Gradient Killing: Convert to string to hide distance
    if cycle_pos == 99:
        
        # 2. Inner Plateau (The "Hour" check)
        # Offset inside the block = timestamp % 100000
        # Slot Size = 100000 // 100
        # Slot Index = Offset // Slot Size
        # Target: 50 (Middle slot)
        
        offset = timestamp % 100000
        slot_size = 100000 // 100
        slot_idx = offset // slot_size
        
        if slot_idx == 50:
            return 0.0  # Success!
        
        # Inner Plateau Penalty (Wrong Hour)
        return 50.0
            
    # Outer Plateau Penalty (Wrong Day)
    return 100.0
