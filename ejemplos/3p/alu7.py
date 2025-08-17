import random


def es_entero_y_positivo(num):
    try:
        aux = int(num)
        if aux <= 0:
            return False
    except ValueError:
        return False
    return True


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


def leer_txt(ruta):
    res = []

    try:
        with open(ruta, "r") as archivo:

            contenido = archivo.readlines()
            for i in range(len(contenido)):
                res.append(
                    split_custom(contenido[i], "\n")
                )  # lo paso a lista de listas

    except FileNotFoundError:  # si el archivo no existe, lo creo

        with open("inventario.txt", "w") as archivo:

            archivo.write("1000: 100" + "\n")
            archivo.write("1001: 100" + "\n")
            archivo.write("1002: 100" + "\n")
            archivo.write("1003: 100" + "\n")
            archivo.write("1004: 100" + "\n")
            archivo.write("1005: 100" + "\n")
            archivo.write("1006: 100" + "\n")
            archivo.write("1007: 100" + "\n")
            archivo.write("1008: 100" + "\n")
            archivo.write("1009: 100" + "\n")

    return res


def vender_productos(ruta, cantidad):

    contenido = leer_txt(ruta)
    print(contenido)
    if not es_entero_y_positivo(cantidad):
        print("La cantidad ingresada debe ser un numero entero positivo.")

    productos_disponibles = []

    for i in range(len(contenido)):  # busco los productos disponibles
        productos_y_cant = contenido[i]

        sep = split_custom(productos_y_cant[0], ":")

        productos = sep[0]

        if int(productos) > 0:
            productos_disponibles.append(productos)

    num_random = random.randint(0, len(productos_disponibles))
    producto_random = productos_disponibles[num_random]
    cantidad_random_a_vender = random.randint(1, 5)


vender_productos("inventario.txt", 1)
