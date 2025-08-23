def pals_unicas(lista):
    pals_unicas = []
    for pal in lista:
        if pal not in pals_unicas:
            pals_unicas.append(pal)
    return pals_unicas


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


def ordenar(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] < lista[j]:
                lista[i], lista[j] = lista[j], lista[i]
    return lista


def promedio(lista):
    total = 0
    for valor in lista:
        total += valor
    return total / len(lista)


def validar(a):
    valido = True
    i = 0
    while i < len(a):
        if not (
            (48 <= ord(a[i]) <= 57)
            or (ord(a[i]) == 32)
            or (65 <= ord(a[i]) <= 90)
            or (97 <= ord(a[i]) <= 122)
            or (ord(a[i]) == 92)
        ):
            print(False)
            valido = False
            return False

        else:
            i += 1

    if valido:
        print(True)
        return True


validar(a)


if validar(a):

    def mostrar_lineas(a):
        j = 0
        while j < len(a) - 1:
            if a[j] == " " and a[j] == a[j + 1]:
                a[j] = ""
                j += 1
            else:
                j += 1
        texto = split(a, "\n")
        contador = 0
        for i in texto:
            contador += 1
            if not (texto[i] == ""):
                print(f"{contador}: {texto[i]},end")

    def promedio_numeros(a):
        i = 0
        total = []
        j = 1
        while i < len(a):
            if 48 <= ord(a[i]) <= 57:
                numero = int(a[i])
                while 48 <= ord(a[i + j]) <= 57:
                    numero = int(a[i] + a[i + j])
                    j += 1
                total.append(int(numero))
            i += 1
        promedio = promedio(total)
        return print(f"El promedio de los números en el texto es {promedio}")

    def sin_repetir(a):
        with open("sin_repeticiones.txt", "w") as archivo:
            palabras = pals_unicas(a)
            palabras_long = []
            for i in range(len(palabras)):
                if not palabras[i] in "0123456789":
                    palabra_str = str(palabras[i])
                    palabras_long.append(len(palabra_str))

            longitudes = ordenar(palabras_long)
            archivo.write(longitudes)

else:
    print("Texto invalido")
    lista[j] = lista[j], lista[i]
    return lista


def promedio(lista):
    total = 0
    for valor in lista:
        total += valor
    return total / len(lista)


def validar(a):
    valido = True
    i = 0
    while i < len(a):
        if not (
            (48 <= ord(a[i]) <= 57)
            or (ord(a[i]) == 32)
            or (65 <= ord(a[i]) <= 90)
            or (97 <= ord(a[i]) <= 122)
            or (ord(a[i]) == 92)
        ):
            print(False)
            valido = False
            return False

        else:
            i += 1

    if valido:
        print(True)
        return True


validar(a)


if validar(a):

    def mostrar_lineas(a):
        j = 0
        while j < len(a) - 1:
            if a[j] == " " and a[j] == a[j + 1]:
                a[j] = ""
                j += 1
            else:
                j += 1
        texto = split(a, "\n")
        contador = 0
        for i in texto:
            contador += 1
            if not (texto[i] == ""):
                print(f"{contador}: {texto[i]},end")

    def promedio_numeros(a):
        i = 0
        total = []
        j = 1
        while i < len(a):
            if 48 <= ord(a[i]) <= 57:
                numero = int(a[i])
                while 48 <= ord(a[i + j]) <= 57:
                    numero = int(a[i] + a[i + j])
                    j += 1
                total.append(int(numero))
            i += 1
        promedio = promedio(total)
        return print(f"El promedio de los números en el texto es {promedio}")

    def sin_repetir(a):
        with open("sin_repeticiones.txt", "w") as textos:
            palabras = pals_unicas(a)
            palabras_long = []
            for i in range(len(palabras)):
                if not palabras[i] in "0123456789":
                    palabra_str = str(palabras[i])
                    palabras_long.append(len(palabra_str))

            longitudes = ordenar(palabras_long)
            textos.write(longitudes)

else:
    print("Texto invalido")
