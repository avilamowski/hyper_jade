def validar(texto):
    i = 0
    valido = True
    if texto == "":
        return False
    while i < len(texto):
        if (65 <= ord(texto[i]) <= 90) or (97 <= ord(texto[i]) <= 122):
            i += 1
        elif 48 <= ord(texto[i]) <= 57:
            i += 1
        elif ((texto[i]) == "\n") or texto[i] == " ":
            i += 1
        else:
            return False
    return valido


validacion = validar("texto a validar")


print(validacion)
