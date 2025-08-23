texto = "Hola gente del universo 123 Chau todos 90 adios"


def validacion(texto):
    val = texto
    if val == " ":
        return False
    for i in val:
        if i not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 ":
            return False
        else:
            return True
    return val
    return texto


def mostrar_lineas(val, texto):
    if val == True:
        text = [texto]
        palabra = []
        for i in text:
            if i == " ":
                palabra.append(texto[:i])
                print(palabra)


val = validacion(texto)
mostrar_lineas(val, texto)
