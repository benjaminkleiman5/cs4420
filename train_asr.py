import numpy as np
import pickle
import json
import kaldiio
import random
import os
import dnn
import ctc
import wang_ctc
import input_generator


#Define initial model
din = 1245 #83*15 
dout = 29
num_hidden_layers = 2
hidden_layer_width= 500
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)
flag = False

with open("asr_model.pkl",'rb') as f:
    network.restore_model("asr_model.pkl")
    f.close()

def train():
    #Info for updating model later
    learning_rate = 0.001
    beta = 0.99
    momentum_w = [np.zeros_like(w) for w in network.weights]
    momentum_b = [np.zeros_like(b) for b in network.biases]
    #Load minibatch of utterances     
    train = input_generator.InputGenerator('train_data.json', batch_size = 5, shuffle = True, context_length = 7, subsampling_rate = 3)
    while train.epoch < 5:
        triplet_batch = train.next()
        #print(str(triplet_batch))
        #features sequence
        x = [triplet[1] for triplet in triplet_batch]
        #print(x)
        #label sequence
        y = [triplet[2] for triplet in triplet_batch]
        #Pad utterance inputs 
        #Calculate maximum input length
        inp_max = max(len(inp) for inp in x)
        padded_inputs = []
        input_loss_mask = []
        #Pad inputs
        for seq in x:
            padded_inputs.append(np.pad(seq, ((0, inp_max - seq.shape[0]), (0,0)), mode='constant'))
            #print("pad shape: " + str(np.pad(seq, ((0, inp_max - seq.shape[0]), (0,0)), mode='constant').shape))
            #Calculate input loss mask 
            input_loss_mask.append(np.pad([1] * len(seq), (0, inp_max - seq.shape[0]), mode='constant'))
        input_loss_mask = np.stack(input_loss_mask)
        flattened_input_loss_mask = input_loss_mask.reshape(-1)
        padded_inputs = np.stack(padded_inputs)
        #print("loss_mask shape: " + str(input_loss_mask.shape))
        #print("flattened_input_loss_mask: " + str(flattened_input_loss_mask.shape))
        #print("padded_inputs: " + str(padded_inputs.shape))
        #Pad utterance outputs     
        #Calculate maximum input length 
        out_max = max(len(out) for out in y)
        padded_outputs = []
        output_loss_mask = []
        #Pad outputs
        for seq in y:
            padded_outputs.append(np.pad(seq, (0, out_max - len(seq)), mode='constant'))
            output_loss_mask.append(np.pad([1] * len(seq), (0, out_max - len(seq)), mode='constant'))
        padded_outputs = np.stack(padded_outputs)
        #Forward the current DNN model
        #For the input we need to reshape and flatten the inputs because right now they are of form (batch by T' by D' but we need it to be flattened to b*T' by D'
        #print("padded_outputs: " + str(padded_outputs.shape))
        flattened_padded_inputs = padded_inputs.reshape(-1, padded_inputs.shape[-1])
        #print("flattened_inputs: " +str(flattened_padded_inputs.shape))
        #forward input is din, batch so we need to transpose
        transposed = flattened_padded_inputs.T
        #print("tranposed_and_flat_inputs: " + str(transposed.shape))
        flattened_padded_outputs = padded_outputs.reshape(-1, padded_outputs.shape[-1])
        out, hidden = network.forward(transposed)
         
        #Need to reshape/unflatten the output of the DNN to (b,T,C)
        #print("out: " + str(out.shape))
        transpose_out = out.T
        #print("tranpose out: " + str(transpose_out.shape))
        unflattened_output = transpose_out.reshape(padded_inputs.shape[0], inp_max, out.shape[0])
        #print("out_unflattened: " + str(unflattened_output.shape))

        #Calculate -log loss
        loss = -np.log(dnn.compute_softmax(transpose_out))
        #print("loss: " + str(loss.shape))
        unflattened_loss = loss.reshape(padded_inputs.shape[0], inp_max, out.shape[0])
        #print("unflattened_loss: " + str(unflattened_loss.shape))
        #print("input lengths: " + str(np.array([len(triplet[1]) for triplet in triplet_batch])))
        #print("output lengths: " + str(np.array([len(triplet[2]) for triplet in triplet_batch])))
        #Obtain forced alignment
        best_costs, paths = wang_ctc.compute_forced_alignment(unflattened_loss, np.array([len(triplet[1]) for triplet in triplet_batch]),
                                                         padded_outputs, np.array([len(triplet[2]) for triplet in triplet_batch]))
        #print("paths: " + str(paths))
        #compute the cross entropy loss on all of the frames because we have the ground truth from the input generator and we are given the best estimate by the forward
        #print("loss_mask: " + str(input_loss_mask))
        flattened_paths = paths.reshape(-1)
        loss, loss_grad = dnn.compute_ce_loss(out, flattened_paths, flattened_input_loss_mask)
        print("ce_loss: " + str(loss))
        
        
        #Use forced alignment as ground truth for calculating loss and gradient descent
        #The input would be the same as the forward for the forward
        grad_w, grad_b = network.backward(transposed, hidden, loss_grad, flattened_input_loss_mask)
        
        #Update DNN model
        momentum_w = [beta * mw + gw for (mw, gw) in zip (momentum_w, grad_w)]
        momentum_b = [beta * mb + gb for (mb, gb) in zip (momentum_b, grad_b)]
        w_updates = [-learning_rate * mw for mw in momentum_w]
        b_updates = [-learning_rate * mb for mb in momentum_b]
        network.update_model(w_updates, b_updates)


        #Store model every 100 steps
        if train.total_num_steps % 200 == 0:
            print(train.total_num_steps)
            network.save_model("asr_model.pkl")


       
    #Save model after epoch finished
    print("epoch completed")
    network.save_model("asr_model.pkl")
    print(f'epoch {train.epoch}')
    
    
    #Print loss to see if it reduces by 0.6 per frame
    
    #You can use the forced alignment test file to see if you are getting a reasonable alignment

if __name__=="__main__":
    train()

