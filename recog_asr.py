import editdistance
import numpy as np
from dnn import FeedForwardNetwork
from input_generator import InputGenerator
from wang_ctc import compute_softmax, beam_search
from utils import pad_sequences


context_length = 7
subsampling_rate = 3
din = 83 * (2 * context_length + 1)
dout = 29
num_hidden_layers = 2
hidden_layer_width = 500
network = FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)
network.restore_model('asr_model.pkl')
dev = InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=context_length, subsampling_rate=subsampling_rate)
total_edits = 0
total_chars = 0

while True:
    batch = dev.next()
    print(batch[0][0])
    if dev.total_num_steps == 2703:
        break
    uttid, x, y = batch[0]
    T_in, d = x.shape
    expanded_x = np.expand_dims(x, axis=0) 
    batch_size = 1  
    tranposed_x = expanded_x.T  
    reshaped_x = tranposed_x.reshape((d, batch_size * T_in))  
    out, hidden = network.forward(reshaped_x)
    reshaped_out = out.T.reshape(batch_size, T_in, dout)  
    prob = compute_softmax(reshaped_out)
    log_prob = np.log(prob + 1e-13)
    input_lens = np.array([T_in])
    hypothesis = beam_search(log_prob, input_lens)
    hypo_string = dev.tokenizer.IdsToString(hypothesis[0])
    ref_string = dev.tokenizer.IdsToString(y)
    edit_distance = editdistance.eval(hypo_string, ref_string)
    total_edits = edit_distance + total_edits
    total_chars = len(ref_string) + total_chars
cer = (total_edits / total_chars) if (total_chars > 0) else 0.0
print(f'CER on dev: {cer:.2%}')


