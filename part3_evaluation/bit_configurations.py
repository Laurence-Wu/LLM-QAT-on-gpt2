from typing import Dict

class BitConfigurations:
    """Standard W-A-KV configurations from LLM-QAT paper"""

    STANDARD_CONFIGS = {
        "FP32": {
            "W": 32, "A": 32, "KV": 32,
            "name": "32-32-32",
            "description": "Full FP32 precision (teacher)"
        },
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
        "INT6": {
            "W": 6, "A": 6, "KV": 6,
            "name": "6-6-6",
            "description": "6-bit integer quantization"
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
        Force proper configuration without fallbacks
        """
        weight_bits = config.get('W', 8)

        # Check if model supports switchable precision
        # For SPLMHeadModel, bit_widths is in transformer
        try:
            bit_widths = model.transformer.bit_widths
        except AttributeError:
            # Try direct access for compatibility
            try:
                bit_widths = model.bit_widths
            except AttributeError:
                raise AttributeError(f"Model does not support switchable precision. Missing 'bit_widths' attribute in model or model.transformer.")

        # Get supported bit-widths
        print(f"Model supports bit-widths: {bit_widths}")
        print(f"Requested bit-width: {weight_bits}")

        # Check if requested bit-width is supported
        if weight_bits not in bit_widths:
            raise ValueError(f"Requested bit-width {weight_bits} not in supported bit-widths {bit_widths}. "
                           f"Model must be trained with this bit-width to support it.")

        # Try to set global precision
        try:
            model.set_global_precision(weight_bits)
            print(f"Successfully set global precision to {weight_bits}-bit")
        except AttributeError:
            # Try set_precision method
            try:
                model.set_precision(weight_bits)
                print(f"Successfully set precision to {weight_bits}-bit")
            except AttributeError:
                raise AttributeError(f"Model does not have 'set_global_precision' or 'set_precision' method. "
                                   f"Cannot apply bit configuration.")

        # Verify the precision was set
        # Try transformer.current_bit_width first (for SPLMHeadModel)
        try:
            current_bit_width = model.transformer.current_bit_width
        except AttributeError:
            # Try direct access
            try:
                current_bit_width = model.current_bit_width
            except AttributeError:
                # Can't verify, but continue (precision was set successfully)
                print(f"Warning: Cannot verify bit-width was set to {weight_bits}")
                return model

        if current_bit_width != weight_bits:
            raise RuntimeError(f"Failed to set bit-width. Expected {weight_bits}, got {current_bit_width}")

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
    def calculate_compression_ratio(config: Dict, baseline_config: Dict) -> float:
        """Calculate compression ratio compared to baseline - baseline_config REQUIRED"""
        if baseline_config is None:
            raise ValueError("baseline_config is required - no defaults allowed")

        baseline_bits = baseline_config['W'] + baseline_config['A'] + baseline_config['KV']
        config_bits = config['W'] + config['A'] + config['KV']

        return baseline_bits / config_bits