from random import *


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


def elige_azar(lista_productos):
    indice = randint(0, 10)
    cantidad_azar = randint(1, 6)
    elegido = lista_productos[indice]

    return elegido, cantidad_azar


def actualiza_listas(lista_productos, lista_cantidades, elegido, cantidad_azar):

    negativo = -elegido

    for i in range(len(lista_productos)):
        if elegido == lista_productos[i]:
            lista_cantidades[i] -= cantidad_azar

    if negativo not in lista_productos:
        lista_productos.append(negativo)
        lista_cantidades.append(cantidad_azar)

    if negativo in lista_productos:
        for i in range(len(lista_productos)):
            if lista_productos[i] == negativo:
                lista_cantidades[i] += cantidad_azar

    return lista_productos, lista_cantidades


def vender_productos(nombre, cantidad):

    lista_productos = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
    lista_cantidades = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

    try:
        cantidad = int(cantidad)
    except ValueError:
        print("La cantidad debe ser un numero entero")
        return

    if cantidad < 0:
        print("La cantidad debe ser positiva")
        return

    comprueba = str(cantidad)
    if "." in comprueba:
        print("La cantidad no puede ser un numero decimal.")
        return

    try:
        with open(nombre, "r") as archivo_txt:
            for linea in archivo_txt:
                auxiliar = split_custom(linea, ":")
                lista_productos.append(int(strip_custom(auxiliar[0])))
                lista_cantidades.append(int(auxiliar[1]))

    except FileNotFoundError:
        print("El archivo seleccionado no existe, se creara uno.")

    for i in range(cantidad):

        elegido, cantidad_azar = elige_azar(lista_productos)
        lista_productos, lista_cantidades = actualiza_listas(
            lista_productos, lista_cantidades, elegido, cantidad_azar
        )
        print(lista_productos)

    with open(nombre, "w") as archivo_txt:
        for i in range(len(lista_productos)):
            archivo_txt.write(
                "{0}: {1}\n".format(lista_productos[i], lista_cantidades[i])
            )

    return


vender_productos("inventario.txt", 2)
