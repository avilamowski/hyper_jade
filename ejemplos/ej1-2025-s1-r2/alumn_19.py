def es_numero(valor):
    try:
        float(valor)
    except Exception as e:
        return False

    return True


def split_custom(texto, separador=" "):  # separo en general la entrada
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


def validar(texto):
    for i in texto:
        if i != "":
            return True
        elif "a" <= i <= "z" or "A" <= i <= "Z":
            return True
        elif es_numero:
            return True
        elif i == " ":
            return True
        elif i == "\n":
            return True
        else:
            return False


def mostrar_lineas(texto):
    texto_separado = split_custom(texto, "\n")

    linea = 1
    if validar:
        for (
            i
        ) in (
            texto_separado
        ):  # si no es espacio va al else, aumenta linea pero no printea
            if i != " ":
                print(f"{linea}: {i}")
                linea += 1
            else:
                linea += 1
    else:
        print("Texto inválido")


def promedio_numeros(texto):
    texto_separado = split_custom(texto, "\n")
    numeros = 0
    total_numeros = 0
    if validar:
        for i in texto_separado:
            if es_numero:
                numero = float(i)
                numeros += numero
                total_numeros += 1

        if total_numeros > 0:
            promedio = numeros / total_numeros
            print(f"El promedio de  los números en el texto es {promedio}")
    else:
        print("Texto inválido")

    return numeros


def ordenar_custom(lista):  # ordenar de mayor a menor longitud
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if len(lista[i]) < len(lista[j]):
                lista[i], lista[j] = lista[j], lista[i]
    return lista


def unicas(texto):  # busco unicas para despues usarlo en sin_repetir
    lista_unicas = []
    texto_separado = split_custom(texto, " ")
    for palabra in texto_separado:
        if palabra not in lista_unicas:
            lista_unicas.append(palabra)
    return lista_unicas


def sin_repetir(texto):
    lista_unicas = unicas(texto)
    lista_ordenada = ordenar_custom(lista_unicas)
    if validar:
        with open("sin_repeticiones.txt", "w", encoding="utf-8") as archivo:
            for palabra in lista_ordenada:
                if not ("0" <= palabra <= "9"):
                    archivo.write(palabra + "\n")
