import random


def leer_inventario(ruta):
    try:
        with open(ruta, "r") as archivo:
            contenido = archivo.read()
            return contenido
    except FileNotFoundError:
        print("El archivo de la ruta ingresada no se pudo ingresar, no existe")
        return False


def no_existe_ruta():
    contenido = []
    for i in range(1000, 1010):
        contenido.append([i, "100"])
    for j in contenido:
        print(f"{j[0]}: {j[1]} \n")


def enlistar_inventario(ruta):
    contenido = leer_inventario(ruta)
    inventario = []
    producto = []
    palabra = ""
    for i in contenido:
        if i == "\n":
            producto.append(palabra)
            inventario.append(producto)
            producto = []
            palabra = ""
        elif i == ":":
            producto.append(palabra)
            palabra = ""
        elif i != " ":
            palabra += i
    if palabra:
        producto.append(palabra)
        inventario.append(producto)
    return inventario


def validar_cantidad(cantidad):
    for i in cantidad:
        if i not in (
            "0123456789"
        ):  # de haber un signo menos,o un punto\coma tambien devuelve
            print("Se deben ingresar numeros enteros y cantidad positiva")
            return False
    return True


def obtener_productos_con_stock(ruta):
    inventario = enlistar_inventario(ruta)
    stock = []
    for i in inventario:
        if i[0][0] != "-":
            if int(i[1]) > 0:
                stock.append(i)
    return stock


def obtener_historial(ruta):
    inventario = enlistar_inventario(ruta)
    historial = []
    for i in inventario:
        if i[0][0] == "-":
            historial.append([i[0][1:], i[1]])
    return historial


""" 
def randoms(ruta, cantidad):
  stock = obtener_productos_con_stock(ruta)
  cdad = int(cantidad)
  producto = random.randint(1000,1009)
  if cdad > 5:
    se_venden = random.randint(1,5)
  else:
    se_venden = random.randint(1,cdad)
  return producto , se_venden
"""


def generar_producto_random(ruta):
    stock = obtener_productos_con_stock(ruta)
    producto = random.randint(1000, 1009)
    return producto


def generar_cantidad(ruta, cantidad):
    cdad = int(cantidad)
    if cdad > 5:
        se_venden = random.randint(1, 5)
    else:
        se_venden = random.randint(1, cdad)
    return se_venden


def modificar_historial(ruta, codigo, cdad):
    historial = obtener_historial(ruta)
    i = 0
    while i < len(historial):
        if historial[i][0] == str(codigo):
            historial[i][1] = str(int(historial[i][1] + cdad))
        i += 1
    return historial


def modificar_stock(ruta, codigo, cdad):
    stock = obtener_productos_con_stock(ruta)
    i = 0
    while i < len(stock):
        if stock[i][0] == str(codigo):
            stock[i][1] = str(int(stock[i][1] - cdad))
        i += 1
    return stock


def venta(ruta, cantidad):
    stock = obtener_productos_con_stock(ruta)
    historial = obtener_historial(ruta)
    cdad = int(cantidad)
    while cdad > 0:
        producto = generar_producto_random(ruta)
        se_venden = generar_cantidad(ruta, cantidad)
        print(f"se vende {producto}: {se_venden} unidades")
        stock = modificar_stock(ruta, producto, se_venden)
        historal = modificar_historial(ruta, producto, se_venden)
        cdad = -se_venden
    return stock, historial


def rescribir_arhivo(ruta, cantidad):
    stock, historial = venta(ruta, cantidad)
    with open(ruta, w) as archivo:
        for i in stock:
            archivo.write(f"{i[0]}: {i[1]}")
        for i in historial:
            arhivo.write(f"-{i[0]}: i[1]")


def vender_productos(ruta, cantidad):
    if not validar_cantidad(cantidad):
        return False
    else:
        stock, historial = venta(ruta, cantidad)
        rescribir_arhivo(ruta, cantidad)


ruta = "inventario.txt"
# print(type(random.randint(1,3)))
cantidad = input("ingrese la cantidad a vender: ")
# print(leer_inventario(ruta))
vender_productos(ruta, cantidad)
