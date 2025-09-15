import torch
from transformers import GPT2Tokenizer
from shared.dataset import create_dataloaders

def test_dataset(dataset_type, train_split, val_split):
    """Test a specific dataset type."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_type.upper()} Dataset")
    print('='*60)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        dataset_type=dataset_type,
        train_split=train_split,
        val_split=val_split,
        batch_size=2,
        max_length=256,
        doc_stride=128,
        num_workers=0
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Attention Mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {batch['labels'].shape}")

    # Show an example
    example_input = batch['input_ids'][0]
    example_labels = batch['labels'][0]

    # Find non-masked labels
    valid_labels = example_labels[example_labels != -100]
    print(f"\nNumber of tokens for loss computation: {len(valid_labels)}")

    # Decode sample
    decoded_text = tokenizer.decode(example_input, skip_special_tokens=True)
    print(f"\nSample text (first 300 chars):")
    print(decoded_text[:300])

    if dataset_type == 'squad':
        # For SQuAD, show where answer starts
        answer_start = (example_labels != -100).nonzero()[0].item() if (example_labels != -100).any() else -1
        if answer_start != -1:
            answer_tokens = example_input[answer_start:]
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            print(f"\nAnswer portion: {answer_text[:100]}")

    return train_loader, val_loader


def main():
    print("Testing Multi-Dataset Support for GPT-2 Training")
    print("="*60)

    # Test SQuAD dataset
    squad_train, squad_val = test_dataset(
        'squad',
        'train[:100]',  # Use subset for testing
        'validation[:50]'
    )

    # Test WikiText-2 dataset (smaller, faster for testing)
    wikitext2_train, wikitext2_val = test_dataset(
        'wikitext-2',
        'train',
        'validation'
    )

    # Test WikiText-103 dataset (uncomment if you want to test the larger dataset)
    # wikitext103_train, wikitext103_val = test_dataset(
    #     'wikitext-103',
    #     'train',
    #     'validation'
    # )

    print("\n" + "="*60)
    print("✓ All datasets loaded successfully!")
    print("="*60)

    # Demonstrate usage in training loop
    print("\nExample training loop structure:")
    print("-"*40)
    print("""
# In your training script:
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        # ... optimizer step, etc.
    """)


if __name__ == "__main__":
    main()