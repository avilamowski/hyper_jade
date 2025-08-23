def es_numero(caracter):
    if "0" <= caracter <= "9":
        return True
    else:
        return False


def es_letra(caracter):
    if "A" <= caracter <= "Z" or "a" <= caracter <= "z":
        return True
    else:
        return False


def es_espacio(caracter):
    if caracter == " ":
        return True
    else:
        return False


def es_enter(caracter):
    if caracter == "\n":
        return True
    else:
        return False


def validar(texto):
    for caracter in texto:
        if not (
            es_numero(caracter)
            or es_letra(caracter)
            or es_espacio(caracter)
            or es_enter(caracter)
        ):
            return False
    return True


def strip(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


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


def mostrar_lineas(texto):
    if validar(texto):
        lineas = split(texto, "\n")
        orden = []
        cant = 0
        while cant < len(lineas):
            cant += 1
            orden.append(str(cant))
        for i in range(len(lineas)):
            linea = lineas[i]
            valida = False
            for j in linea:
                if es_letra(j) or es_numero(j):
                    valida = True
            if linea != "" and valida:
                line = strip(linea)
                print(f"{orden[i]}: {line}")
    else:
        print("Texto invalido")


def digitos_numericos(texto):
    suma = 0
    cant_nums = 0
    lineas = split(texto, "\n")
    for i in range(len(lineas)):
        palabras = split(lineas[i], " ")
        for j in range(len(palabras)):
            palabra = palabras[j]
            number = True
            for caracter in palabra:
                if not es_numero(caracter):
                    number = False
            if number:
                suma += int(palabra)
                cant_nums += 1
    return suma, cant_nums


def promedio_numeros(texto):
    if validar(texto):
        suma, cant_nums = digitos_numericos(texto)
        try:
            promedio = suma / cant_nums
            print(f"El promedio de los numeros en el texto es de {promedio}")
        except ZeroDivisionError:
            print(f"El texto no posee numeros.")
    else:
        print("Texto invalido")


def ordenar_descendente(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if len(lista[i]) < len(lista[j]):
                lista[i], lista[j] = lista[j], lista[i]
    return lista


def toda_palabra(texto):
    todas = []
    lineas = split(texto, "\n")
    for i in range(len(lineas)):
        palabras = split(lineas[i], " ")
        for j in range(len(palabras)):
            palabra = palabras[j]
            palabra = palabras[j]
            number = True
            for caracter in palabra:
                if not es_numero(caracter):
                    number = False
            if not number and palabra != "":
                if palabra not in todas:
                    todas.append(palabra)
    return todas


def sin_repetir(texto):
    if validar(texto):
        todas = toda_palabra(texto)
        ordenadas = ordenar_descendente(todas)
        with open("sin_repeticiones.txt", "w", encoding="utf-8") as archivo:
            for i in ordenadas:
                archivo.write(i)
    else:
        print("Texto invalido")
