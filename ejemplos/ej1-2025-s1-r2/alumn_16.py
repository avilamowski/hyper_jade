def validar(texto):
    i = 0
    validar = True
    while i < len(texto) and validar:
        if texto[i] == "":
            validar = False

        elif not ("a" <= texto[i] <= "z" or "A" <= texto[i] <= "Z" or texto[i] == " "):
            validar = False

        elif texto[i] == "\n":
            validar = True

        else:
            validar = True
        i += 1

    if validar:
        print("True")
        return True
    else:
        print("False")
        return False


def mostrar_lineas(texto):
    cont = 0
    nueva = ""
    i = 0
    while i < len(texto):
        if texto[i] != "\n" and texto[i] != "":
            nueva += texto[i]

        else:
            cont += 1
            if texto[i] == "\n" and texto[i + 1] != "\n" and texto[i] != "":
                print(f"{cont}: {nueva}")
                nueva = ""

            else:
                nueva = ""
        i += 1
    cont += 1
    if nueva != "":
        print(f"{cont}: {nueva}")


def crear_texto(texto):
    lista_unica = []
    lista_long = []
    palabra = ""
    long = 0
    i = 0
    while i < len(texto):
        if texto[i] != " " or texto[i] != "\n":
            palabra += texto[i]
        else:
            long = len(palabra)
            if palabra not in lista_unica:
                lista_unica.append(palabra)
                lista_long.append(long)
                long = 0
                palabra == ""
            else:
                palabra == ""
                long = 0
        i += 1
    crear_archivo(lista_long, lista_unica)


def crear_archivo(long, unica):
    for i in range(len(long) - 1):
        for j in range(i + 1, len(long)):
            if long[i] < long[j]:
                long[i], long[j] = long[j], long[i]
                unica[i], unica[j] = unica[j], unica[i]
    try:
        with open("sin_repeticiones.txt", "w", encoding="utf-8", newline="") as archivo:
            for palabra in unica:
                archivo.write(palabra + "\n")
    except FileNotFoundError:
        print(f"El archivo no existe")


# Tengo error cuando valida '\n', el resto de la validacion funciona.
# Se generan listas vacias, no encuentro donde esta el error.


def main():
    texto = "Hola     "
    ingreso = validar(texto)
    if ingreso is not False:
        mostrar_lineas(texto)
        crear_texto(texto)
    else:
        print("Texto inválido")
        return None


main()
