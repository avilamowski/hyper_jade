def validar(texto):
    if texto == "":
        return False
    for i in range(len(texto)):
        if texto[i] not in "abcdefghijklmnopqrstuvwxyz 0123456789":
            if i != (len(texto) - 1):
                if texto[i : i + 2] == "\n":
                    return True
            else:
                return False
    return True


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


def mostrar_lineas(texto):
    if validar(texto):
        sin_saltos = split_custom(texto, "\n")

        for i in range(len(sin_saltos)):
            contar_invalido = 0

            for g in range(len(sin_saltos[i])):
                if sin_saltos[i][g] == " " or sin_saltos[i][g] == "":
                    contar_invalido += 1

            if contar_invalido != len(sin_saltos[i]):
                sin_saltos[i] = split_custom(sin_saltos[i], " ")

                g = 0

                for w in range(len(sin_saltos[i])):
                    while g < len(sin_saltos[i][w]) and g != " ":
                        g += 1
                    if g == " ":
                        sin_saltos[i][w][g + 1 :]
                        g = 0

                resultado = " "

                for w in range(len(sin_saltos[i])):
                    if w != (len(sin_saltos[i]) - 1):
                        resultado += sin_saltos[i][w] + " "
                    else:
                        resultado += sin_saltos[i][w]

                sin_saltos[i] = resultado

                print(str(i + 1) + ":" + sin_saltos[i])

        return sin_saltos

    else:
        print("Texto inválido")
        return False


def promedio(texto):
    if validar(texto):
        sin_saltos = split_custom(texto, "\n")
        promedio = 0
        cuenta_numeros = 0

        for i in range(len(sin_saltos)):
            contar_invalido = 0

            for g in range(len(sin_saltos[i])):
                if sin_saltos[i][g] == " " or sin_saltos[i][g] == "":
                    contar_invalido += 1

            if contar_invalido != len(sin_saltos[i]):
                sin_saltos[i] = split_custom(sin_saltos[i], " ")

                g = 0

                for w in range(len(sin_saltos[i])):
                    while g < len(sin_saltos[i][w]) and g != " ":
                        g += 1
                    if g == " ":
                        sin_saltos[i][w][g + 1 :]
                        g = 0

                nada = 0

                for w in range(len(sin_saltos[i])):
                    try:
                        sin_saltos[i][w] = int(sin_saltos[i][w])
                        promedio += sin_saltos[i][w]
                        cuenta_numeros += 1
                    except ValueError:
                        nada += 1

        print(f"El promedio de los números en el texto es {promedio/cuenta_numeros}")
        return promedio / cuenta_numeros

    else:
        print("Texto inválido")
        return False
