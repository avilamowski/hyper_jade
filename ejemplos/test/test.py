def show_results(names):
    # Remove duplicates
    unique_names = sorted(set(names), key=lambda x: x.lower())
    
    print("=== Sorted Names ===")
    for name in unique_names:
        print(name)
    
    # Length frequencies
    length_freq = {}
    for name in unique_names:
        l = len(name)
        length_freq[l] = length_freq.get(l, 0) + 1
    
    print("\n=== Length Frequencies ===")
    for length, count in sorted(length_freq.items()):
        print(f"Length {length}: {count}")
    
    # Character frequencies (case-insensitive)
    char_freq = {}
    for name in unique_names:
        for ch in name.lower():
            char_freq[ch] = char_freq.get(ch, 0) + 1
    
    print("\n=== Character Frequencies ===")
    for ch, count in sorted(char_freq.items()):
        print(f"'{ch}': {count}")


def main():
    names = []
    print("Enter names (press Enter on empty input to finish):")
    while True:
        inp = input().strip()
        if not inp:
            break
        if inp.isalpha():
            names.append(inp)
        else:
            print("Invalid input, only alphabetic characters allowed.")
    
    show_results(names)


if __name__ == "__main__":
    main()
