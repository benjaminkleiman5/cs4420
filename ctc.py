import numpy as np

def compute_forced_alignment(logits, input_lens, labels, output_lens):
    # Set up all labels for calculation
    padded = []
    label_len = len(labels)
    lenIter = range(label_len)

    # Make best_cost_array of correct shape, ensuring dtype is float32
    best_cost_array = np.zeros((label_len, 2*len(labels[0]) + 1), dtype=np.float32)
    
    for x in lenIter:
        # Take relevant part of labels
        padded.append(labels[x][:output_lens[x]])
        # Pad with zeroes
        padded[x] = [0] + [item for sublist in zip(labels[x], [0]*label_len) for item in sublist]
        # Set initial cost values
        best_cost_array[x] = np.float32(10**30)  # ensure value is float32
        best_cost_array[x][0] = logits[x][0][padded[x][0]]
        best_cost_array[x][1] = logits[x][0][padded[x][1]]
    
    # Make prev_node_array for each label, ensuring dtype is float32
    prev_node_array = np.zeros((np.max(input_lens), 2*len(labels[0]) + 1), dtype=np.float32)
    prev_node_arrays = np.repeat(prev_node_array[np.newaxis, :], label_len, axis=0)
    
    for t in range(np.max(input_lens)):
        for x in lenIter:
            twoLplus1 = 2 * output_lens[x] + 1
            # Skip if t exceeds input length
            if t + 1 >= input_lens[x]:
                continue
            
            # Create a flat copy of best_cost_array[x] for comparison
            flat = np.copy(best_cost_array[x])
            for y in range(twoLplus1):
                print("x: " + str(x))
                print("y: " + str(y))
                print("t+!: " + str(t+1))
                flat[y] = flat[y] + logits[x][t + 1][padded[x][y]]
            
            # Create diag array with correct cost values, ensure dtype is float32
            diag = np.concatenate((
                [np.float32(10**30)], 
                best_cost_array[x][:twoLplus1-1], 
                np.full((2 * np.max(output_lens) + 1) - twoLplus1, np.float32(10**30))
            ), dtype=np.float32)
            
            for y in range(1, twoLplus1):
                diag[y] = diag[y] + logits[x][t+1][padded[x][y]]
            
            # Use np.minimum to get the min cost array
            mini = np.minimum(flat, diag)
            different_indices = np.where(flat > mini)
            
            # Update prev_node_array based on differences between flat and diag
            for index in range(twoLplus1):
                if index in different_indices[0]:
                    prev_node_arrays[x][t+1][index] = 1
                else:
                    prev_node_arrays[x][t+1][index] = 0
            
            # Create diag2 ensuring dtype is float32
            diag2 = np.concatenate((
                [np.float32(10**30), np.float32(10**30)], 
                best_cost_array[x][:twoLplus1-2], 
                np.full((2 * np.max(output_lens) + 1) - twoLplus1, np.float32(10**30))
            ), dtype=np.float32)
            diag2copy = np.copy(diag2)
            protected = []
            
            for entry in range(2, len(diag2)):
                if((padded[x][entry] == 0) or (padded[x][entry-2] == 0) or (padded[x][entry] == (padded[x][entry-2]))):
                    diag2[entry] = np.float32(10**30)
                    diag2[entry-2] = np.float32(10**30)
                else:
                    diag2[entry] = diag2copy[entry] + logits[x][t+1][padded[x][entry]]
                    if entry - 2 not in protected:
                        diag2[entry-2] = np.float32(10**30)
                    protected.append(entry)
            
            # Determine optimal moves using diag2
            mini2 = np.minimum(diag2, mini)
            different_indices2 = np.where(mini > mini2)
            
            # Update prev_node_array based on diag2
            for index in different_indices2[0]:
                prev_node_arrays[x][t+1][index] = 2
            
            # Update best_cost_array
            best_cost_array[x] = np.copy(mini2)    
    best_costs = []
    paths = np.zeros((label_len, np.max(input_lens)), dtype=np.float32)
    for x in range(label_len):
        twoLplus1 = 2*output_lens[x] + 1
        last = best_cost_array[x][twoLplus1 - 1]
        second_to_last = best_cost_array[x][twoLplus1 - 2]
        current_node = 0
        if last < second_to_last:
            best_costs.append(last)
            y = input_lens[x] - 1
            current_node = twoLplus1 - 1
            while y > 0:
                paths[x][y] = padded[x][int(current_node)]
                current_node = current_node - prev_node_arrays[x][y][int(current_node)]
                y = y-1
            paths[x][0] = padded[x][int(current_node)]
        else:
            best_costs.append(second_to_last)
            y = input_lens[x] - 1
            current_node = twoLplus1 - 2
            while y > 0:
                paths[x][y] = padded[x][int(current_node)]
                current_node = current_node - prev_node_arrays[x][y][int(current_node)]
                y = y-1
            paths[x][0] = padded[x][int(current_node)]

    return best_costs, paths
            
        
