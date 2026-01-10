import csv


def limpiar_datos_txt(venta_txt):
    try:
        with open(venta_txt, "r", encoding="utf-8") as f_txt:
            datos_txt = f_txt.read()
        if datos_txt[-1] != "\n":
            datos_txt += "\n"
        venta_lista_cant = []
        venta_producto = []
        venta_cantidad = []
        dato_temporario = ""
        for letra in datos_txt:
            if letra == "\n":
                venta_lista_cant += [dato_temporario]
                dato_temporario = ""
            else:
                dato_temporario += letra
        dato_temporario_producto = ""
        for pedido in venta_lista_cant:
            for i in range(len(pedido)):
                if "a" <= pedido[i] <= "z":
                    dato_temporario_producto += pedido[i]
                if pedido[i] == ":":
                    venta_producto += [dato_temporario_producto]
                    dato_temporario_producto = ""
                if "0" <= pedido[i] <= "9":
                    venta_cantidad += [int(pedido[i])]
        return venta_producto, venta_cantidad
    except:
        return "0.0", "0.0"


def limpiar_datos_csv(inventario_csv):
    producto_inventario = []
    precio_inventario = []
    cant_inventario = []
    with open(inventario_csv, "r", encoding="utf - 8") as f_csv:
        lector = csv.reader(f_csv)
        for linea in lector:
            producto_inventario += [linea[0]]
            precio_inventario += [linea[1]]
            cant_inventario += [linea[2]]
    return producto_inventario, precio_inventario, cant_inventario


# def ignorar_no_existentes():


def costo(inventario_csv, venta_txt):
    venta_producto, venta_cantidad = limpiar_datos_txt(venta_txt)
    if venta_producto == "0.0":
        return "0.0"
    else:
        producto_inventario, precio_inventario, cant_inventario = limpiar_datos_csv(
            inventario_csv
        )
        costo_total_vendidos = 0
        return costo_total_vendidos


def hacer_pedido(inventario_csv, venta_txt, cantidad_final):
    return lista_productos_en_inventario


def main():
    inventario_csv = "inventario.csv"
    venta_txt = "venta.txt"
    costo_total_vendidos = costo(inventario_csv, venta_txt)


main()
