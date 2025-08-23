def validar(text):
    if text:
        for i in range(len(text)):
            if not (
                ("A" <= text[i] <= "Z")
                or ("a" <= text[i] <= "z")
                or ("0" <= text[i] <= "9")
                or (text[i] == " ")
            ):
                if not (text[i] == "\n"):
                    print("hola")
                    return False
        return True
    else:
        return False


# -----------------
texto = input("texto: ")
vari = validar(texto)
