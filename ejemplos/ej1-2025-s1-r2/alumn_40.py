def validar(texto):
    if texto == "":
        return False
    try:
        letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        numero = "0123456789"
        continuar = True

        while continuar == True:
            for caracter in texto:
                if caracter in numero:
                    continuar = True
                elif caracter in letras:
                    continuar = True
                elif caracter == " ":
                    continuar = True
                elif "\n" in texto:
                    continuar = True
                else:
                    continuar = False
                    return False
            return True

    except ValueError:
        return False
    except Exception as e:
        return e


# 2.mostrar_lineas(texto): (1,5 ptos.) que reciba un texto
# como parámetro y, si el texto es válido,
# muestre cada línea no vacía del texto en líneas separadas, precedidas por su número de línea original.


# Requisitos:


# Las líneas vacías (que no contienen ningún carácter visible)
# no deben mostrarse.
##El número de línea que se muestra
# corresponde a la posición original de la línea en el texto, considerando todas las líneas, incluidas las vacías.
# En cada línea mostrada,
# las palabras deben separarse por un único espacio, sin importar cuántos espacios haya en el texto original.
# No debe agregarse espacio extra al final de la línea mostrada.}


def split_custom(texto, separador=" "):
    resultado = []
    pos = []
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
    if validar(texto) == False:
        print("Texto Invalido")
        return
    else:
        split_custom(texto, "\n")

        return "1:" + texto
