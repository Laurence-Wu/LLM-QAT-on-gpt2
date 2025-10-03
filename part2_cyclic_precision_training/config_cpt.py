class ModelConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1
        self.bit_widths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 32]


        self.shared_lora_rank = 16
        self.shared_lora_alpha = 32
        self.quantizer_type = 'log'
        self.quantizer_per_bit = {
            2: 'log', 3: 'log', 4: 'log', 5: 'log', 6: 'log', 7: 'log',
            8: 'log', 9: 'log', 10: 'log', 11: 'log', 12: 'log', 13: 'log',
            14: 'log', 15: 'log', 16: 'log', 17: 'log', 18: 'log', 32: None
        }
        self.use_per_channel = True
        self.gradient_bits = 8
        self.activation_bits_per_bit = {
            2: 40, 3: 36, 4: 32, 5: 28, 6: 24, 7: 20, 8: 16, 9: 12, 10: 10,
            11: 10, 12: 10, 13: 10, 14: 8, 15: 8, 16: 8, 17: 4, 18: 4, 32: 0
        }
        self.weight_gradient_bits = 16
        self.activation_gradient_bits = 8
        self.use_gradient_checkpointing = True

class CPTConfig:
    def __init__(self):
        self.total_cycles = 15
        self.schedule_type = 'cosine'
        self.prt_start_bits = 2
        self.prt_threshold = 0.01
        self.prt_iterations = 50

class TrainingConfig:
    def __init__(self):
        self.train_split = 'train[:80000]'
        self.val_split = 'validation[:5000]'
        self.batch_size = 32
        self.max_seq_length = 256
        self.doc_stride = 128
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        self.max_grad_norm = 1.0
        self.num_epochs = 150
        self.gradient_accumulation_steps = 3
        self.target_bits = 5
        self.eval_interval = 50
        self.empty_cache_interval = 25
        self.num_workers = 0
        self.device = 'cuda'
        self.fp16 = False
        self.log_interval = 10
        self.verbose = True

def get_config():
    return {
        'model': ModelConfig(),
        'cpt': CPTConfig(),
        'training': TrainingConfig(),
    }
