import tokenizer


def main():
    """Dummy test code to make sure things work as we expect"""
    custom_tokenizer = tokenizer.MyTokenizer()
    
    load_from_file = False
    if load_from_file:
        custom_tokenizer.load("shakespeare.model")
    else:    
        with open("data/shakespeare.txt", "r") as file:
            text = file.read()
        custom_tokenizer.train(text, 256 + 20)

    encoded = custom_tokenizer.encode("To be or not to be, that is the question.")
    print(encoded)

    decoded = custom_tokenizer.decode(encoded)
    print(decoded)

    # Save the tokenizer
    custom_tokenizer.save("shakespeare")


if __name__ == "__main__":
    main()