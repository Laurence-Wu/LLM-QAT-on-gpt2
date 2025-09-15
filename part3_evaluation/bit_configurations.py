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
        "INT4": {
            "W": 4, "A": 4, "KV": 4,
            "name": "4-4-4",
            "description": "4-bit integer quantization"
        },
        "INT2": {
            "W": 2, "A": 2, "KV": 2,
            "name": "2-2-2",
            "description": "2-bit integer quantization (extreme)"
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
        For switchable models, we only set the weight precision
        since the model handles activation and KV cache internally
        """
        weight_bits = config.get('W', 8)

        # For switchable models, we set the precision if it's supported
        if hasattr(model, 'set_global_precision'):
            # Check if the requested bit-width is supported
            if hasattr(model, 'bit_widths') and weight_bits in model.bit_widths:
                model.set_global_precision(weight_bits)
            else:
                # Find the closest supported bit-width
                if hasattr(model, 'bit_widths'):
                    supported = model.bit_widths
                    closest = min(supported, key=lambda x: abs(x - weight_bits))
                    print(f"Warning: {weight_bits}-bit not supported, using {closest}-bit instead")
                    model.set_global_precision(closest)
                else:
                    model.set_global_precision(weight_bits)
        elif hasattr(model, 'set_precision'):
            model.set_precision(weight_bits)

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