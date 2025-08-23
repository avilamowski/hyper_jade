def split_custom(texto, separador=" "):
    palabras = []
    palabra = ""
    for char in texto:
        if char == separador:
            if palabra:
                resultado.append(palabra)
                palabra = ""
        else:
            palabra += char
    if palabra:
        palabras.append(palabra)
    return palabras


def strip_custom(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


# 1
def validar(texto):
    for char in texto:
        if not (
            ("a" <= char <= "z")
            or ("A" <= char <= "Z")
            or ("0" <= char <= "9")
            or char == " "
            or char == "\n"
        ):
            return False
    return True


# 2
def mostrar_lineas(texto):
    if not validar(texto):
        print("Texto no valido.")
        return

    lineas = split_custom(texto, "\n")
    linea_numero = 1
    for linea in lineas:
        if strip_custom(linea):
            palabras = split_custom(linea)
            linea_nueva = ""
            for i in range(len(palabras)):
                if i > 0:
                    linea_nueva += " "
                linea_nueva += palabras[i]
            print(f"{linea_numero}.{linea_nueva}")
        linea_numero += 1


# 3
def promedio_numeros(texto):
    if not validar(texto):
        print("Texto no valido")
        return

    numeros = []
    palabras = split_custom(texto)
    for palabra in palabras:
        es_numero = True
        for char in palabra:
            if char < "0" or char > "9":
                es_numero = False
            if es_numero:
                numeros.append(int(palabra))

        if numeros:
            promedio = 0
            for n in numeros:
                promedio += n
                promedio /= len(numeros)
                print(f"Promedio de los numeros:{promedio:2.f}")
        else:
            print("No hay numeros en el texto")


# 4
def sin_repetir(texto):
    if not validar(texto):
        print("Texto no valido")
        return
    palabras_unicas = []

    palabras_totales = split_custom(texto)
    for palabra in palabras_totales:
        es_numero = True
        for char in palabra:
            if char < "0" or char > "9":
                es_numero = False
        if not es_numero and palabra not in palabras_unicas:
            palabras_unicas.append(palabra)
    for i in range(len(palabras_unicas)):
        for j in range(i + 1, len(palabras_unicas)):
            if len(palabras_unicas[i] < len(palabras_unicas[j])):
                palabras_unicas[i], palabras_unicas[j] = (
                    palabras_unicas[j],
                    palabras_unicas[i],
                )

    with open("sin_repeticiones.txt", "w") as archivo:
        for palabra in palabras_unicas:
            archivo.write(palabra + "\n")
    print("Archivo creado")
