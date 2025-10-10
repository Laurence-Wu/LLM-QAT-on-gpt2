

class ModelConfig:
    def __init__(self):
        # ===== GPT-2 Architecture (same as part1) =====
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.layer_norm_epsilon = 1e-5
        self.embd_pdrop = 0.1

        # ===== Quantization Configuration (same as part1) =====
        self.quantization_bits = 8
        self.activation_bits = 8
        self.quantizer_type = 'minmax'

        # Quantizer per bit-width
        self.quantizer_per_bit = {
            3: 'minmax',
            4: 'minmax',
            5: 'log',
            6: 'log',
            7: 'log',
            8: 'log',
            9: 'log',
            10: 'log',
            11: 'log',
            12: 'log',
            13: 'log',
            14: 'log',
            15: 'log',
            16: 'log',
            32: None
        }

        # LoRA Configuration (same as part1)
        self.lora_rank = 16
        self.lora_alpha = 32
        self.bit_widths = [7, 32]  # Train 7-bit with 32-bit teacher

        self.teacher_bits = 32

        # LoRA per bit-width
        self.lora_rank_per_bit = {
            3: 64, 4: 64, 5: 64, 6: 64, 7: 64, 8: 64,
            9: 64, 10: 64, 11: 64, 12: 64, 13: 64, 14: 64,
            15: 64, 16: 64, 32: 0
        }
        self.lora_alpha_per_bit = {
            3: 64, 4: 64, 5: 64, 6: 64, 7: 64, 8: 64,
            9: 64, 10: 64, 11: 64, 12: 64, 13: 64, 14: 64,
            15: 64, 16: 64, 32: 0
        }

        # Activation bits per bit-width
        self.activation_bits_per_bit = {
            3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
            9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
            15: 15, 16: 16, 32: 32
        }

        self.per_channel_quantization = True


class TrainingConfig:
    def __init__(self):
        # ===== Dataset Configuration (SQuAD-specific) =====
        self.dataset_name = 'squad'  # or 'squad_v2' for SQuAD 2.0
        self.train_split = 'train'
        self.val_split = 'validation'

        # SQuAD-specific parameters
        self.max_seq_length = 384  # Maximum sequence length
        self.doc_stride = 128  # Sliding window stride
        self.max_query_length = 64  # Maximum question length
        self.max_answer_length = 30  # Maximum answer length
        self.n_best_size = 20  # For beam search answer extraction

        # ===== Training Configuration =====
        self.batch_size = 16  # Reduced from 32 due to longer sequences
        self.learning_rate = 3e-5  # Lower LR for QA fine-tuning
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.adam_betas = (0.9, 0.999)
        self.max_grad_norm = 1.0

        self.num_iterations = 1000  # Adjust based on dataset size
        self.gradient_accumulation_steps = 8
        self.eval_interval = 50
        self.empty_cache_interval = 25
        self.num_workers = 0

        # ===== Distillation Configuration (same as part1) =====
        self.distill_alpha_kl = 1.0  # Weight for KL divergence loss
        self.distill_alpha_feature = 1e-7  # Weight for feature distillation
        self.distill_temperature = 3.0  # Temperature for distillation
        self.teacher_update_interval = 10
        self.distill_warmup_steps = 100
        self.feature_layers = None  # None = all layers
        self.cache_size = 32  # Teacher output cache size
