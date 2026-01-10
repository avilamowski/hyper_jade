import csv


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


def leer_txt(venta):
    try:
        with open(venta, "r") as archivo:
            fila = []
            for linea in archivo:
                if linea[:-1] == "\n":
                    linea = linea[:-1]
                else:
                    fila.append(linea)
            vendido = []
            cant = []
            for i in fila:
                num = ""
                palabra = ""
                for j in i:
                    if "0" <= j <= "9":
                        num += j
                    elif "a" <= j <= "z":
                        palabra += j
                cant.append(num)
                vendido.append(palabra)
            # print(vendido, cant)
            return vendido, cant
    except FileNotFoundError:
        print("error")
        return 0.0


def leer_csv(inventario):
    try:
        invent = []
        costo = []
        unidades = []

        with open(inventario, "r") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                invent.append(fila[0])
                costo.append(fila[1])
                unidades.append(fila[2])
            # print(invent, costo, unidades)
            return invent, costo, unidades
    except FileNotFoundError:
        print("error")


def costo(archivo1, archivo2, total, pedidos, cuanto):
    invent, valor, unidades = leer_csv(archivo1)
    print(invent, valor, unidades)
    pedido = input("pedido: ")
    vendido = 0
    if pedido in invent:
        done = False
        while not done:
            try:
                cantidad = int(input("cantidad: "))
                done = True
            except ValueError:
                print("numero entero")
        for i in range(len(invent)):
            if pedido == invent[i]:
                pedidos.append(pedido)
                if cantidad >= int(unidades[i]):
                    cuanto.append(unidades[i])
                    vendido = int(unidades[i]) * float(valor[i])
                if cantidad < int(unidades[i]):
                    cuanto.append(cantidad)
                    vendido = cantidad * float(valor[i])
        else:
            vendido += 0
    total += vendido
    print(total, pedidos, cuanto)
    with open(archivo2, "w") as archivo:
        for i in range(len(pedidos)):
            archivo.write(f"{pedidos[i]}: {cuanto[i]}\n")
    return pedidos, cuanto


# hacer_pedido(inventario_csv, venta_txt, cantidad_final)
# → retorna una lista con los nombres de los productos existentes
# en el inventario que se deben reponer para llegar a la
# cantidad_final (int) deseada después de haber hecho la venta.


# Considerar que:


# Si la cantidad final es menor a la cantidad existente
# en el inventario para todos los productos, se retorna una lista vacía.
# Ejemplo: Había 10 gomas, vendí una y la cantidad final es 5,
# entonces no tengo que comprar gomas.


# Si el archivo de venta no existe, se retorna una lista pedida
# en base al inventario solamente.
# Si un producto del pedido no existe en el inventario, se ignora.
# cantidad_final es siempre mayor a cero. No validar.


def hacer_pedido(archivo1, archivo2, ideal):
    invent, valor, unidades = leer_csv(archivo1)

    vendido = 0
    pedidos, cuantos = leer_txt(archivo2)
    print(pedidos, cuantos)
    for i in range(len(pedidos)):
        for j in range(len(invent)):
            if pedidos[i] == invent[j]:
                suma = int(unidades[j]) - int(cuanto[i])
                if suma < ideal:
                    print(pedidos[i])

    if not leer_txt(archivo2):
        print(invent)
        return invent

    with open(archivo1, "w") as archivo:
        escritor = csv.writer(archivo)
        for i in range(len(invent)):
            if invent[i] in pedidos:
                escritor.writerow([invent[i], valor[i], suma])
            else:
                escritor.writerow([invent[i], valor[i], unidades[i]])


# leer_txt('ventas.txt')
leer_csv("inventario.csv")
done = False
total = 0
pedidos = []
cuanto = []
while not done:
    sigo = input("compra?")
    if sigo == "si":
        costo("inventario.csv", "ventas.txt", total, pedidos, cuanto)
    else:
        done = True


done = False
while not done:
    try:
        ideal = int(input("ideal: "))
        done = True
    except ValueError:
        print("numero entero")


hacer_pedido("inventario.csv", "ventas.txt", ideal)
##mi ejeercico cumple con los ejemplos dados creo haberlo=e peusto try excpet a todos los with open, quiza me olivde pero no tengo mas tiempo.
