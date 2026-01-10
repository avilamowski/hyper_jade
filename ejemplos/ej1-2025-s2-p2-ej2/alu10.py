import csv


def invent(FILE):

    FILE = "inventario.csv"
    objetos = []
    cantidad = []
    precio = []

    try:
        with open(FILE, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                objetos.append(fila[0])
                cantidad.append(fila[1])
                precio.append(fila[2])
    except FileNotFoundError:
        print("El archivo", FILE, "no existe.")
    else:
        print(objetos)
        print(cantidad)
        print(precio)

    objetos = []
    cantidad = []
    precio = []


def venta(fn):

    pedido = ""
    cant = ""
    try:
        fn = "venta.txt"
        fd = None
        fd = open(fn, "r")
        contenido = fd.read()
        print(contenido)
        contador = 0
        for i in range(len(contenido)):
            if contenido[i] == ":":
                contador += 1
                cant = cant + ","
                pedido = pedido + ","
            elif contenido[i] != ":" and contador % 2 == 0:
                peidido = pedido + contenido[i]
            elif contenido[i] != ":" and contador % 2 == 0:
                cant = cant + contenido[i]

        print(pedido)
        print(cant)

        pedidos = [pedido]
        cants = [cant]
        File = "pedi.csv"
        with open(File, "a", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            for i in range(len(pedidos)):
                escritor.writerow([pedidos[i], cants[i]])

    except FileNotFoundError:
        print("El archivo no existe.")
    else:
        print(pedidos)
        print(cants)
    finally:
        if fd:
            fd.close()


def costo():
    invent("inventario.csv")
    venta("venta.txt")
    for i in range(len(pedidos)):
        if pedidos[i] == objetos[i]:
            recaudado = recaudado + float(cants) * floats(precio)
            cantidad[i] = floats(cantidad[i]) - float(cants[i])
        print(recaudado)


costo()
