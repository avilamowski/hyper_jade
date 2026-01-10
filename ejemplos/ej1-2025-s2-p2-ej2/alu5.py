def split_custom(texto, separador=":"):
    resultado = []
    palabra = ""
    for char in texto:
        if char == separador or char == "\n":
            if palabra:
                resultado.append(palabra)
                palabra = ""
        else:
            palabra += char
    if palabra:
        resultado.append(palabra)
    return int(resultado)


def costo(archivo_csv, txt):
    try:
        ### csv ###
        with open(archivo_csv, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            productos = []
            precios = []
            cantidades = []
            for fila in lector:
                productos.append(fila[0])
                precios.append(fila[1])
                cantidades.append(strip_custom(fila[2]))

        ### txt ###

        fd = None
        fd = open(txt, "r")
        contenido = fd.read()
        fd.close()
        ###
        lista_palabras = split_custom(contenido)

        productos_venta = []
        cantidades_venta = []
        for i in range(len(lista_palabras)):
            lp = lista_palabras[i]
            if i % 2 == 0:
                productos_venta.append(lp)
            else:
                cantidades_venta.append(strip_custom(lp))

        cantidades_venta_limpia = []
        productos_venta_limpia = []
        for i in range(len(cantidades_venta)):
            if productos_venta[i] not in productos_venta_limpia:
                productos_venta_limpia.append(productos_venta[i])
                cantidades_venta_limpia.append(cantidades_venta[i])
            else:
                k = 0
                while (
                    k < len(productos_venta_limpia)
                    and productos_venta_limpia[k] != productos_venta[i]
                ):
                    k += 1
                cantidades_venta_limpia[k] += 1

        ### empieza funcionalidad de funcion ###
        productos_vendidos = []
        precios_vendidos = []
        for i in range(len(productos_venta_limpia)):
            pvl = productos_venta_limpia[i]
            if pvl in productos and pvl not in productos_vendidos:
                productos_vendidos.append(pvl)
                precios_vendidos.append(precios[i])

        cantidades_se_vendieron = []
        for j in range(len(cantidades_venta_limpia)):
            cvl = cantidades_venta_limpia[j]
            if cvl <= cantidades[j]:
                cantidades_se_vendieron.append(cvl)
            else:
                cantidades_se_vendieron.append(cantidades[j])

        ### parte del costo ###

        costo_final = 0
        o = 0
        while o < len(productos_vendidos):
            costo_final += cantidades_se_vendieron[o] * precios_vendidos[o]
            o += 1

        print(costo_final)
        return costo_final

    except FileNotFoundError or ValueError or IndexError or NameError:
        print("Hubo un error")


costo("inventario.csv", "ventas.txt")
