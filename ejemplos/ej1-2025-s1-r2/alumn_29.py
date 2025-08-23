def validar(texto):
    if texto == "":
        return False
    else:
        i = 0
        while i < len(texto):
            if not ("A" <= texto[i] <= "Z"):
                return False
            if not ("0" <= texto[i] <= "9"):
                return False
            if texto[i] != " ":
                return False
            if texto[i] != "\n":
                return False
            i += 1
        return True


def mostrar_lineas(texto):
    if validar(texto) != True:
        print("El texto es invalido")
    else:
        i = 0
        inicio = 0
        while i < len(texto):
            if texto[i] == "\n":
                palabra = texto[incio:i]
                inicio = i
                if palabra != "":
                    print(palabra)
            i += 1
            if i == len(texto):
                palabra = texto[i:]
                print(palabra)
