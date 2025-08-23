def strip_custom(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


def split_custom(texto, separador):
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
    es_valido = True
    if texto == "":
        return False
        es_valido = False
    else:
        for i in range(len(texto)):
            if not (
                (texto[i] >= "a" and texto[i] <= "z")
                or (texto[i] >= "A" and texto[i] <= "Z")
                or (texto[i] == " ")
                or (texto[i] >= "0" and texto[i] <= "9")
                or (texto[i : i + 1] == "\n")
            ):
                es_valido = False
        if es_valido:
            return True
    return False


def mostrar_lineas(texto):
    contador = 0
    frase = ""
    if validar(texto) == True:

        for i in range(len(texto)):
            if not (texto[i : i + 1] == "\n"):
                frase += texto[i]
            else:
                contador += 1
                if len(frase) > 1:
                    print(str(contador) + ":" + strip_custom(frase) + "\n")
                frase = ""

        contador += 1
        if len(frase) > 0:
            print(str(contador) + ":" + strip_custom(frase) + "\n")

    else:
        return "Texto inválido"


def promedio_numeros(texto):
    suma_numeros = 0
    numero = ""
    numeros = []
    if validar(texto) == True:
        for i in range(len(texto)):
            if texto[i] >= "0" and texto[i] <= "9":
                numero += texto[i]
            else:
                if numero:
                    suma_numeros += int(numero)
                    numeros.append(numero)
                    numero = ""
            if len(numeros) != 0:
                promedio = suma_numeros / len(numeros)
    else:
        return "texto invalido"
    return promedio


def ordenar_longitud(lista):
    n = len(lista)
    for i in range(n):
        for j in range(0, n - i - 1):
            if len(lista[j]) < len(lista[j + 1]):
                lista[j], lista[j + 1] = lista[j + 1], lista[j]
    return lista


def sin_repetir(texto):
    if validar(texto) == True:
        with open("sin_repeticiones.txt", "w", encoding="utf-8") as file:
            palabras = split_custom(texto, " ")

            palabras_ordenadas = ordenar_longitud(palabras)

            for palabra in palabras_ordenadas:
                es_valido = True
                contador = 0
                for i in range(len(palabra)):
                    if palabra[i] >= "0" and palabra[i] <= "9":
                        contador += 1
                if contador == len(palabra):
                    es_valido = False
                if es_valido:
                    if palabra != " " or palabra != "":
                        file.write(palabra + "\n")
    else:
        return "texto invalido"
