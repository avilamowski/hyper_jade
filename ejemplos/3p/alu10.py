from random import randint


def split_custom(texto, separador):
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


def validar_cantidad(cantidad):
    if type(cantidad) != int:
        return False
    elif cantidad <= 0:
        return False
    if type(cantidad) == int and cantidad > 0:
        return True


def formato(lista):
    resultado = []
    for i in range(len(lista)):
        palabra = "-"
        palabra += lista[i]
        resultado.append(palabra)
    return resultado


def buscar_indice(lista, elemento):
    for i in range(len(lista)):
        if elemento == lista[i]:
            return i


def venta(cantidad, productos, inventario):
    i = 0
    vendido = []
    cantidad_vendida = []
    while i < cantidad:
        elegido = randint(0, len(inventario) - 1)
        vendidos = randint(1, 5)

        if vendidos > inventario[elegido]:
            if productos[elegido] in vendido:
                indice = buscar_indice(vendido, productos[elegido])
                cantidad_vendida[indice] += inventario[elegido]
            else:
                vendido.append(productos[elegido])
                cantidad_vendida.append(inventario[elegido])
            inventario[elegido] -= inventario[elegido]
        else:
            if productos[elegido] in vendido:
                indice = buscar_indice(vendido, productos[elegido])
                cantidad_vendida[indice] += vendidos
            else:
                vendido.append(productos[elegido])
                cantidad_vendida.append(vendidos)
            inventario[elegido] -= vendidos
        i += 1
    resultado = formato(vendido)
    return resultado, cantidad_vendida


def abrir_txt(archivo_txt):
    palabras = []
    try:
        with open(archivo_txt, "r", encoding="UTF-8") as archivo:
            contenido = archivo.read()
            palabras = split_custom(contenido, "\n")
            print("Se abrio correctamente")
            return palabras
    except FileNotFoundError:
        return palabras


def crear_listas():
    productos = []
    inventario = []
    j = 1000
    for i in range(10):
        productos.append(str(j))
        j += 1
        inventario.append(100)
    return productos, inventario


def funcion(palabras):
    producto = []
    stock = []
    historial = []
    historial_venta = []
    for i in range(len(palabras)):
        splitteado = split_custom(palabras[i], ":")
        if splitteado[0][0] != "-":
            producto.append(splitteado[0])
            stock.append(splitteado[1])
        else:
            historial.append(splitteado[0])
            historial_venta.append(splitteado[1])
    return producto, stock, historial, historial_venta


def vender_productos(ruta, cantidad):
    valido = validar_cantidad(cantidad)
    if not valido:
        raise Exception("La cantidad debe ser un entero positivo.")
    else:
        palabras = abrir_txt(ruta)
        # len==0 no existe el archivo o no tiene nada, lo creo
        if len(palabras) == 0:
            productos, stock = crear_listas()
            vendido, cantidad_vendida = venta(cantidad, productos, stock)
            with open(ruta, "w", encoding="UTF-8") as archivo:
                for i in range(len(productos)):
                    archivo.write(f"{productos[i]}:{stock[i]}\n")
                for j in range(len(vendido)):
                    archivo.write(f"{vendido[j]}:{cantidad_vendida[j]}\n")
            print("Archivo creado")


try:
    # vender_productos("inventario.txt", -3)
    # este hace que entre en el except y muestre el mensaje de "La cantidad debe ser un entero positivo."
    # sin que aparezca el cartel rojo, solo muestra el mensaje
    vender_productos("inventario.txt", 10)
except Exception as e:
    print(f"Error: {e}")
