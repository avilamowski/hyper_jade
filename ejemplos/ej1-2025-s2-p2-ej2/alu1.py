import csv


def leer_archivo_csv(ruta_archivo_csv):
    nombre = []
    precio_unitario = []
    cantidad_stock = []
    try:
        with open(ruta_archivo_csv, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            for linea_en_partes in lector:
                nombre.append(linea_en_partes[0])
                precio_unitario.append(linea_en_partes[1])
                cantidad_stock.append(linea_en_partes[2])
    except FileNotFoundError:
        print("0.0")
    return nombre, precio_unitario, cantidad_stock


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


def leer_archivo_txt(ruta_archivo_txt):
    try:
        with open(ruta_archivo_txt, "r", encoding="utf-8") as archivo:
            contenido = archivo.read()
    except FileNotFoundError:
        print("0.0")

    ventas = split_custom(contenido, "\n")
    return ventas


def manejar_ventas(lista1):
    producto_venta = []
    cantidad = []
    for i in range(len(lista1)):
        lista = split_custom(lista1[i], ":")
        producto_venta.append(lista[0])
        cantidad.append(lista[1])
    return producto_venta, cantidad


def ordenar_producto_vendido(producto_vendido, cantidad_vendida):
    producto_vendido_unico = []
    cantidad_vendida_unica = []
    for i in range(len(producto_vendido)):
        if producto_vendido[i] not in producto_vendido_unico:
            producto_vendido_unico.append(producto_vendido[i])
            cantidad_vendida_unica.append(cantidad_vendida[i])
        else:
            for k in range(len(producto_vendido_unico)):
                if producto_vendido_unico[k] == producto_vendido[i]:
                    cantidad_vendida_unica[k] = int(cantidad_vendida_unica[k]) + int(
                        cantidad_vendida[i]
                    )
    return producto_vendido_unico, cantidad_vendida_unica


def ordenar_conjunto(producto, precio, disponibles, prod_v, cant_v):
    productos_totales = []
    disponibles_de_ellos = []
    valor_precio = []
    vendidos_de_ellos = []
    for i in range(len(prod_v)):
        for k in range(len(producto)):
            if prod_v[i] == producto[k]:
                productos_totales.append(producto[k])
                disponibles_de_ellos.append(disponibles[k])
                valor_precio.append(precio[k])
                vendidos_de_ellos.append(cant_v[i])
            if prod_v[i] not in producto:
                prod_v[i] = 0
    return productos_totales, disponibles_de_ellos, vendidos_de_ellos, valor_precio


def costo(ruta_archivo_csv, ruta_archivo_txt):
    producto, precio, disponibles = leer_archivo_csv(ruta_archivo_csv)
    vendido = leer_archivo_txt(ruta_archivo_txt)
    producto_vendido, cantidad_vendida = manejar_ventas(vendido)
    prod_v, cant_v = ordenar_producto_vendido(producto_vendido, cantidad_vendida)
    productos1, disponibles1, ventas1, precio1 = ordenar_conjunto(
        producto, precio, disponibles, prod_v, cant_v
    )

    total_comprado = 0
    for i in range(len(productos1)):
        if int(ventas1[i]) > int(disponibles1[i]):
            ventas1[i] = disponibles1[i]
        total_comprado += float(ventas1[i]) * float(precio1[i])
    print(total_comprado)
    return total_comprado


def hacer_pedido(ruta_archivo_csv, ruta_archivo_txt, cantidad_final):
    producto, precio, disponibles = leer_archivo_csv(ruta_archivo_csv)
    vendido = leer_archivo_txt(ruta_archivo_txt)
    producto_vendido, cantidad_vendida = manejar_ventas(vendido)
    prod_v, cant_v = ordenar_producto_vendido(producto_vendido, cantidad_vendida)
    productos1, disponibles1, ventas1, precio1 = ordenar_conjunto(
        producto, precio, disponibles, prod_v, cant_v
    )

    cantidad_luego_de_vender = []
    items_a_reponer = []

    for i in range(len(ventas1)):
        if int(ventas1[i]) > int(disponibles1[i]):
            ventas1[i] = disponibles1[i]

        cantidad_luego_de_vender.append(float(disponibles1[i]) - float(ventas1[i]))
        if int(cantidad_luego_de_vender[i]) < int(cantidad_final):
            items_a_reponer.append(productos1[i])
    print(items_a_reponer)
    return items_a_reponer
