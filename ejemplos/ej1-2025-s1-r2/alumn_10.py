def validar(texto):

    valido = True
    if texto == "":
        valido = False

    for caracter in texto:
        if not (
            "A" <= caracter <= "Z"
            or "a" <= caracter <= "z"
            or "0" <= caracter <= "9"
            or caracter == " "
            or caracter == "\n"
        ):
            valido = False
    return valido


def split_custom(texto, separador=" "):
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


def strip_custom(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


def mostrar_lineas(texto):
    valido = validar(texto)
    if valido:
        lineas = split_custom(texto, "\n")

        for i in range(len(lineas)):
            linea = strip_custom(lineas[i])

            if linea != "":
                palabras = split_custom(linea, " ")
                palabras_limpias = []

                for j in palabras:
                    palabra_limpia = strip_custom(j)

                    if palabra_limpia != "":

                        palabras_limpias.append(palabra_limpia)
                nueva = ""
                if len(palabras_limpias) > 0:
                    nueva = palabras_limpias[0]

                    for c in range(1, len(palabras_limpias)):
                        nueva += " " + palabras_limpias[c]

                print(str(i + 1) + ":" + nueva)


texto = "Hola  gente del universo\n \n123 \n Chau todos\n 90 adios"
mostrar_lineas(texto)
