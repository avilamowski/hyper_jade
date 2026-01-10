inventario = "inventario.csv"
ventas = "ventas.txt"


def leer_csv(file):
    import csv

    productos = []
    precios = []
    cantidad = []

    with open(file, "r", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            # print(fila)
            productos.append(fila[0])
            precios.append(float(fila[1]))
            cantidad.append(int(fila[2]))
    return productos, precios, cantidad


# leer_csv(inventario)


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


def leer_txt(arch):
    try:
        fn = arch
        fd = None
        fd = open(fn, "r")
        contenido = fd.read()
        filas = split_custom(contenido, separador="\n")
        # print(filas)
        return filas
    except FileNotFoundError:
        print("El archivo no existe.")
    finally:
        if fd:
            fd.close()


leer_txt(ventas)


def es_numero(a):
    numeros = "1234567890"
    for i in range(len(a)):
        if a[i] in numeros:
            return True
        else:
            return False


def es_letra(a):
    letras = "abcdefghijklmnopqrstuvwxyz"
    for i in range(len(a)):
        if a[i] in letras:
            return True
        else:
            return False


def separar(arch):
    lista = leer_txt(arch)
    productos = []
    cantidad = []
    for producto in lista:
        palabra = ""
        num = 0
        for i in range(len(producto)):
            if producto[i] != ":" and not es_numero(producto[i]):
                palabra += producto[i]
        productos.append(palabra)

        for i in range(len(producto)):
            if producto[i] != ":" and not es_letra(producto[i]):
                num += int(producto[i])
        cantidad.append(num)

    return productos, cantidad


def costo(inventario, ventas):
    productos_pedidos, cantidad_pedida = separar(ventas)
    productos_disp, precios, cantidad_disp = leer_csv(inventario)
    precio_total = 0
    precio_prod = 0
    hacer_cuenta = True

    for j in range(len(productos_pedidos)):
        for i in range(len(productos_disp)):
            if productos_disp[i] == productos_pedidos[j]:
                cantidad_disp[i] -= cantidad_pedida[j]
                if cantidad_disp[i] >= 0 and hacer_cuenta:
                    precio_prod = cantidad_pedida[j] * precios[i]
                    precio_total += precio_prod
                if cantidad_pedida[j] > cantidad_disp[i] and hacer_cuenta:
                    precio_prod = cantidad_disp[i] * precios[i]
                    if precio_prod >= 0:
                        precio_total = precio_total
                    else:
                        precio_total = precio_total
            elif productos_pedidos[j] not in productos_disp:
                hacer_cuenta = False

    print(f"El precio total del pedido es: ${precio_total}")


costo(inventario, ventas)
