def validar(texto):
    if len(texto) == 0:
        return False

    i = 0
    while i < len(texto):
        c = texto[i]
        codigo = ord(c)
        if not (
            (65 <= codigo <= 90)
            or (97 <= codigo <= 122)
            or (48 <= codigo <= 57)
            or (codigo == 32)
            or (codigo == 10)
        ):
            return False
        i += 1
    return True


def split(texto, separador=" "):
    resultado = []
    palabra = ""
    for char in texto:
        if char == separador:
            if palabra:
                resultado.append(palabra)
                palabra = ""
        else:
            palabra += char
    if palabra:
        resultado.append(palabra)
    return resultado


def strip(s):
    i = 0
    while i < len(s) and (s[i] == " " or s[i] == "\n"):
        i += 1
    s = s[i:][::-1]

    j = 0
    while j < len(s) and (s[j] == " " or s[j] == "\n"):
        j += 1
    s = s[j:][::-1]

    return s


def unico_espacio_entre_palabras(texto):
    nuevo = ""
    palabra = ""
    espacio = False
    for i in range(len(texto)):
        if texto[i] != " ":
            palabra += texto[i]
        else:
            if palabra and not espacio:
                espacio = True
                nuevo += palabra
                nuevo += " "
                palabra = ""
                espacio = False
    if palabra:
        nuevo += palabra
    return strip(nuevo)


def mostrar_lineas(texto):
    OK = False
    if True == validar(texto):
        OK = True
    if OK:
        texto = strip(texto)
        L = split(texto, separador="\n")
        for i in range(len(L)):
            if L[i] != " ":
                L[i] = unico_espacio_entre_palabras(L[i])
                print(f"{i + 1}: {strip(L[i])}")
    else:
        print("Texto inválido")


def palabras_numericas(texto):
    res = []
    numeros = ""
    for i in range(len(texto)):
        if texto[i] != " " and ("0" <= texto[i] <= "9"):
            numeros += texto[i]
        else:
            OK = True
            if numeros != "":
                for j in range(len(numeros)):
                    if (
                        ("a" <= numeros[j] <= "z")
                        or ("A" <= numeros[j] <= "Z")
                        or (numeros[j] == " ")
                    ):
                        OK = False
                if OK:
                    res.append(numeros)
                    numeros = ""
    if numeros:
        res.append(numeros)
    return res[1:]


def promedio_numeros(texto):
    L = palabras_numericas(texto)
    suma = 0
    for i in range(len(L)):
        suma += int(L[i])
    if len(L) != 0:
        promedio = suma / len(L)
    return promedio


def ordenamiento_descendente(L):
    ln = len(L)
    for i in range(ln):
        for j in range(ln - i - 1):
            if int(L[j]) < int(L[j + 1]):
                L[j], L[j + 1] = L[j + 1], L[j]
    return L


def sin_repetir(texto, file):
    OK = False
    if True == validar(texto):
        OK = True
    if OK:
        L_sin_repeticiones = []
        L = split(texto, separador=" ")
        for elem in L:
            if elem not in L_sin_repeticiones:
                L_sin_repeticiones.append(elem)
        L_ordenada = ordenamiento_descendente(L_sin_repeticiones.append)
        try:
            with open(file, "w", newline="", encoding="utf-8") as arch:
                for elem in L_ordenada:
                    arch.write(elem)
        except FileNotFoundError:
            print("Error, no se encontro el archivo")
