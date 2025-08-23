def validar(texto):
    error = 0
    if texto == "":
        error += 1
    for i in range(len(texto)):
        if not (
            "A" < texto[i] < "Z"
            or "a" <= texto[i] <= "z"
            or "0" <= texto[i] <= "9"
            or texto[i] == " "
            or texto[i] == "\n"
        ):
            error += 1
    if error > 0:
        return False
    else:
        return True


def strip_texto(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


def mostrar_lineas(texto):
    validacion = validar(texto)
    if validacion == True:
        oraciones = ""
        i = 0
        contador = 1
        orden = 1
        numero = 0
        while i < len(texto):
            if not texto[i] == "\n":
                oraciones += texto[i]
                if "0" <= texto[i] <= "9":
                    numero += 1
                contador += 1
            if texto[i] == "\n" and oraciones != "":
                print(f"{orden}:{strip_texto(oraciones)}")
                orden += 1
                oraciones = ""
            if texto[i] == "\n" and oraciones == "":
                orden += 1
            i += 1
        print(f"{orden}:{strip_texto(oraciones)}")


def ordenar_texto(texto):
    for i in range(len(texto)):
        for j in range(i + 1, len(texto)):
            if len(texto[i]) > len(texto[j]):
                len(texto[i]), len(texto[j]) == len(texto[j]), len(texto[i])
    return texto


def sin_repetir(texto):
    with open("sin_repeticiones.txt", "w", encoding="utf-8") as file:
        file.write(ordenar_texto(texto))


def main():
    texto = "   Hola     como estas\n\n yo muy\n\n bien   "
    print(validar(texto))
    if validar(texto) == True:
        strip_texto(texto)
        print(mostrar_lineas(texto))
        ordenar_texto(texto)
        sin_repetir(texto)


main()
