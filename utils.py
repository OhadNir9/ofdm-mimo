import numpy as np

def _last_two_distinct_elements_indices(lst):
    if len(lst) < 2:
        return None  # Not enough elements to compare
    
    # Reverse iterate through the list
    for i in range(len(lst) - 1, 0, -1):
        if lst[i] != lst[i - 1] and lst[i] != 0 and lst[i-1] != 0:
            return i, i-1
    
    return None  # All elements are equal    

def diversity_order_from_ser_snr_curve(ser, snr_db):
    first, second = _last_two_distinct_elements_indices(ser)
    print(f"sers: {np.log10(ser[first])}, {np.log10(ser[second])}")
    print(f"snrs: {snr_db[first]}, {snr_db[second]}")
    diversity_order = -10 * (np.log10(ser[first]) - np.log10(ser[second])) / (
        snr_db[first] - snr_db[second]
    )
    #print(f"Diversity Order: {10*diversity_order:.2f}")
    return diversity_order

def generate_n_qubit_combinations(N):
    """
    Generate all possible combinations for N qubits, where each qubit can have values 0-3.
    
    Args:
        N (int): Number of qubits
        
    Returns:
        numpy.ndarray: Array of shape (N, 4^N) containing all possible combinations
    """
    # Create base values for each qubit (0,1,2,3)
    values = np.arange(4)
    
    # Create meshgrid for N qubits
    # We need to call meshgrid with N arguments
    grids = np.meshgrid(*[values] * N, indexing='ij')
    
    # Stack and reshape to get Nx4^N array
    return np.vstack([grid.ravel() for grid in grids])
