import csv


# abro archivo csv
def abro_archivo_csv(file):
    lista_nombre = []
    lista_precio = []
    lista_cant = []
    try:
        with open(file, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                lista_nombre.append(fila[0])
                lista_precio.append(float(fila[1]))
                lista_cant.append(int(fila[2]))
    except FileNotFoundError:
        print("El archivo", file, "no existe.")
    return lista_nombre, lista_precio, lista_cant  # devuelve mis tres listas


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


def abro_archivo_txt(file):
    lista_producto = []
    lista_cantidad = []
    try:
        fn = file
        fd = None
        fd = open(fn, "r")
        contenido = fd.read()
        lista = split_custom(contenido, "\n")
        for i in lista:
            lista_aux = split_custom(i, ":")
            lista_producto.append(lista_aux[0])
            lista_cantidad.append(
                int(lista_aux[1])
            )  # asumo que la cantidad es un entero
    except FileNotFoundError:
        print("El archivo no existe.")
    finally:
        if fd:
            fd.close()
        return lista_producto, lista_cantidad


def buscar_pos(lista, palabra):
    for i in range(len(lista)):
        if lista[i] == palabra:
            return i
    return -1


def lista_unica(lista1, lista2):
    lista_unica_1 = []
    lista_unica_2 = []
    for i in range(len(lista1)):
        if lista1[i] not in lista_unica_1:
            lista_unica_1.append(lista1[i])
            lista_unica_2.append(lista2[i])
        else:
            posicion = buscar_pos(lista_unica_1, lista1[i])
            lista_unica_2[posicion] += lista2[i]
    return lista_unica_1, lista_unica_2


# retorna un float con el costo total de todos los productos que se pudieron vender.
def costo(inventario_csv, venta_txt):  # los dos archivos son
    lista_nombre_csv, lista_precio_csv, lista_cant_csv = abro_archivo_csv(
        inventario_csv
    )
    lista_producto_txt, lista_cantidad_txt = abro_archivo_txt(venta_txt)
    lista_producto_txt, lista_cantidad_txt = lista_unica(
        lista_producto_txt, lista_cantidad_txt
    )
    lista_cant = []
    costo_total = 0
    if lista_producto_txt:
        for i in range(len(lista_nombre_csv)):
            cant = 0
            for j in range(len(lista_producto_txt)):
                if lista_nombre_csv[i] == lista_producto_txt[j]:
                    # primera cond: cantidad sea mayor
                    if lista_cantidad_txt[j] > lista_cant_csv[i]:
                        cant += lista_cant_csv[i]
                        lista_cant.append(0)
                    elif lista_cantidad_txt[j] <= lista_cant_csv[i]:
                        cant += lista_cantidad_txt[j]
                        total = lista_cant_csv[i] - lista_cantidad_txt[j]
                        lista_cant.append(total)
            if not cant:
                lista_cant.append(lista_cant_csv[i])
            costo_total += cant * lista_precio_csv[i]
        return (costo_total, lista_cant)
    else:
        return (float(costo_total), lista_cant)


def hacer_pedido(inventario_csv, venta_txt, cantidad_final):
    lista_nombre_csv, lista_precio_csv, lista_cant_csv = abro_archivo_csv(
        inventario_csv
    )
    lista_producto_txt, lista_cantidad_txt = abro_archivo_txt(venta_txt)
    lista_producto_txt, lista_cantidad_txt = lista_unica(
        lista_producto_txt, lista_cantidad_txt
    )
    lista_reponer = []
    costo_venta, lista_cant_actual = costo(inventario_csv, venta_txt)
    if lista_producto_txt and lista_nombre_csv:
        for i in range(len(lista_nombre_csv)):
            if lista_cant_actual[i] < cantidad_final:
                lista_reponer.append(lista_nombre_csv[i])
        return lista_reponer
    elif not lista_poducto_txt and lista_nombre_csv:
        return lista_reponer
    elif not lista_nombre_csv:
        return None
