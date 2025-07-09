from src.generate import generate_text, load_resources

def main():
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    model, tokenizer = load_resources()

    # Set your prompt and generation settings
    seed_prompt = "She was quite"
    num_words = 50
    top_k = 10
    temperature = 0.9

    print("\n📝 Generating text...\n")
    generated_text = generate_text(
        prompt=seed_prompt,
        tokenizer=tokenizer,
        model=model,
        num_words=num_words,
        top_k=top_k,
        temperature=temperature
    )

    print(generated_text)

if __name__ == "__main__":
    main()
