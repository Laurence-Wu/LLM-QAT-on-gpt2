
class ModelConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1
        self.quantization_bits = 8
        self.activation_bits = 8
        self.quantizer_type = 'minmax'
        self.quantizer_per_bit = {
            3: 'minmax',
            4: 'minmax',
            5: 'minmax',
            6: 'log',
            8: 'log',
            16: 'log',
            24: 'log',
            32: None
        }
        self.lora_rank = 16
        self.lora_alpha = 32
        self.bit_widths = [3, 4, 5, 32]
        self.teacher_bits = 32
        self.lora_rank_per_bit = {3: 64, 4: 64, 5: 64, 32: 0}
        self.lora_alpha_per_bit = {3: 64, 4: 64, 5: 64, 32: 0}
        self.activation_bits_per_bit = {3: 3, 4: 4, 5: 5, 32: 32}
        self.per_channel_quantization = True


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
        self.num_iterations = 550
        self.gradient_accumulation_steps = 8
        self.eval_interval = 50
        self.empty_cache_interval = 25
        self.num_workers = 0
        self.distill_alpha_kl = 1.0
        self.distill_alpha_feature = 1e-7
        self.distill_temperature = 3.0
        self.teacher_update_interval = 10
        self.distill_warmup_steps = 100
        self.feature_layers = None
        self.cache_size = 32
