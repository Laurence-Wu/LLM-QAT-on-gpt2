from typing import Dict

class BitConfigurations:
    """Standard W-A-KV configurations from LLM-QAT paper"""

    STANDARD_CONFIGS = {
        "FP16": {
            "W": 16, "A": 16, "KV": 16,
            "name": "16-16-16",
            "description": "Full precision baseline"
        },
        "INT8": {
            "W": 8, "A": 8, "KV": 8,
            "name": "8-8-8",
            "description": "8-bit integer quantization"
        },
        "W4A8KV8": {
            "W": 4, "A": 8, "KV": 8,
            "name": "4-8-8",
            "description": "4-bit weights, 8-bit activations and KV cache"
        },
        "W4A8KV4": {
            "W": 4, "A": 8, "KV": 4,
            "name": "4-8-4",
            "description": "4-bit weights and KV cache, 8-bit activations"
        },
        "W4A16KV16": {
            "W": 4, "A": 16, "KV": 16,
            "name": "4-16-16",
            "description": "4-bit weights only"
        },
        "W8A8KV4": {
            "W": 8, "A": 8, "KV": 4,
            "name": "8-8-4",
            "description": "8-bit weights and activations, 4-bit KV cache"
        },
        "W4A6KV16": {
            "W": 4, "A": 6, "KV": 16,
            "name": "4-6-16",
            "description": "Mixed precision configuration"
        },
        "W2A16KV16": {
            "W": 2, "A": 16, "KV": 16,
            "name": "2-16-16",
            "description": "Extreme weight quantization"
        },
        "W3A8KV8": {
            "W": 3, "A": 8, "KV": 8,
            "name": "3-8-8",
            "description": "3-bit weight quantization"
        }
    }

    @staticmethod
    def apply_config_to_model(model, config: Dict):
        """
        Apply W-A-KV configuration to switchable model
        Set weight bits, activation bits, and KV cache bits
        """
        weight_bits = config.get('W', 8)
        activation_bits = config.get('A', 8)
        kv_bits = config.get('KV', 8)

        if hasattr(model, 'set_global_precision'):
            model.set_global_precision(weight_bits)
        elif hasattr(model, 'set_layer_precision'):
            n_layers = model.n_layer if hasattr(model, 'n_layer') else model.config.n_layer
            layer_config = [weight_bits] * n_layers
            model.set_layer_precision(layer_config)

        if hasattr(model, 'set_activation_bits'):
            model.set_activation_bits(activation_bits)

        if hasattr(model, 'set_kv_cache_bits'):
            model.set_kv_cache_bits(kv_bits)
        elif hasattr(model, 'h') and len(model.h) > 0:
            for block in model.h:
                if hasattr(block, 'attn') and hasattr(block.attn, 'kv_quantizer'):
                    block.attn.kv_quantizer.set_num_bits(kv_bits)

        return model

    @staticmethod
    def get_config_string(config: Dict) -> str:
        """Get configuration string in W-A-KV format"""
        return f"{config['W']}-{config['A']}-{config['KV']}"

    @staticmethod
    def parse_config_string(config_str: str) -> Dict:
        """Parse configuration string like '4-8-8' to dict"""
        parts = config_str.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid config string: {config_str}. Expected format: W-A-KV (e.g., '4-8-8')")

        return {
            'W': int(parts[0]),
            'A': int(parts[1]),
            'KV': int(parts[2]),
            'name': config_str
        }

    @staticmethod
    def get_all_configs() -> Dict:
        """Return all available configurations"""
        return BitConfigurations.STANDARD_CONFIGS.copy()

    @staticmethod
    def calculate_compression_ratio(config: Dict, baseline_config: Dict = None) -> float:
        """Calculate compression ratio compared to baseline (default FP16)"""
        if baseline_config is None:
            baseline_config = BitConfigurations.STANDARD_CONFIGS['FP16']

        baseline_bits = baseline_config['W'] + baseline_config['A'] + baseline_config['KV']
        config_bits = config['W'] + config['A'] + config['KV']

        return baseline_bits / config_bits