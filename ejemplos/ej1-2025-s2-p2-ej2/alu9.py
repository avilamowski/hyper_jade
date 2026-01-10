import csv


def abrir_manejo_csv_txt(inventario_csv, venta_txt):
    FILE = inventario_csv
    variedad = []
    precio = []
    cantidad = []
    pedido = []
    cant_pedido = []
    try:
        with open(FILE, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                variedad.append(fila[0])
                precio.append(fila[1])
                cantidad.append(int(fila[2]))
    except FileNotFoundError:
        print("El archivo", FILE, "no existe.")
    aux = ""
    try:
        arch = venta_txt
        with open(arch, "r") as archive:
            texto = archive.read()
            for i in range(len(texto)):
                if not texto[i] == ":" or texto[i] == "\n":
                    aux += texto[i]
                if texto[i] == ":":
                    pedido.append(aux)
                    aux = ""
                if texto[i] == "\n":
                    cant_pedido.append(aux[:-1])
                    aux = ""
                if i == (len(texto) - 1):
                    cant_pedido.append(aux)
                    aux = ""
        return variedad, precio, cantidad, cant_pedido, pedido
    except FileNotFoundError:
        print("El archivo", FILE, "no existe.")


def costo(inventario_csv, venta_txt):
    variedad, precio, cantidad, cant_pedido, pedido = abrir_manejo_csv_txt(
        inventario_csv, venta_txt
    )
    costo = 0
    for i in range(len(variedad)):
        for j in range(len(pedido)):
            if variedad[i] == pedido[j]:
                if str(cantidad[i]) >= cant_pedido[j]:
                    costo += (float(precio[i])) * (float(cant_pedido[j]))
                    cantidad[i] -= float(cant_pedido[j])
                else:
                    costo += (float(precio[i])) * (float(cantidad[i]))
    print(costo)


def hacer_pedido(inventario_csv, venta_txt, cantidad_final):
    variedad, precio, cantidad, cant_pedido, pedido = abrir_manejo_csv_txt(
        inventario_csv, venta_txt
    )
    cantidad_total_pedida = []
    for i in range(len(variedad)):
        cont = 0
        for j in range(len(pedido)):
            if variedad[i] == pedido[j]:
                cont += int(cant_pedido[j])
            if j == (len(pedido) - 1):
                cantidad_total_pedida.append(cont)
    pedido_hacer = []
    for i in range(len(cantidad)):
        if (int(cantidad[i]) - cantidad_total_pedida[i]) - cantidad_final < 0:
            pedido_hacer.append(variedad[i])
    print(pedido_hacer)


costo("inventario.csv", "venta.txt")
hacer_pedido("inventario.csv", "venta.txt", 10)
