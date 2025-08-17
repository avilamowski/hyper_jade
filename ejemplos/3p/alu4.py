import random


# funciones permitidas


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


def strip_custom(texto):
    inicio = 0
    fin = len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


# funcion principal


def vender_productos(ruta, cantidad):
    texto = str(cantidad)
    cantidad_entera = True

    for letra in texto:
        if letra < "0" or letra > "9":
            cantidad_entera = False

    if not cantidad_entera == 0 or int(cantidad) <= 0:
        print("La cantidad debe ser un entero positivo.")
        return

    lista = []
    try:
        archivo = open(ruta, "r")
        linea = archivo.readline()
        while linea != "":
            partes = split_custom(linea, ":")
            if len(partes) != 2:
                print("linea con formato incorrecto:", linea)
            else:
                codigo = strip_custom(partes[0])
                valor = strip_custom(partes[1])
                fila = [codigo, valor]
                lista.append(fila)
            linea = archivo.readline()
        archivo.close()
    except Exception as e:
        print("Ocurrio un error al abrir el archivo:", e)
        return

    productos_disponibles = []
    j = 0
    while j < len(lista):
        codigo = lista[j][0]
        if codigo[0] != "":
            stock = int(lista[j][1])
            if stock > 0:
                productos_disponibles.append(codigo)
        j += 1

    vendidas = 0
    intentos = 0
    max_intentos = 100

    while vendidas < int(cantidad) and intetos < max_intentos:
        if len(productos_disponibles) > 0:
            pos = random.randint(0, len(productos_disponibles) - 1)
            codigo = productos_disponibles[pos]
            unidades = random.randint(1, 5)

            j = 0
            while j < len(lista):
                if lista[j][0] == codigo:
                    stock_actual = int(lista[j][1])
                    if vendido <= stock_actual:
                        nuevo_stock = stock_actual - unidades
                        if nuevo_stock < 0:
                            nuevo_stock = 0
                            unidades = stock_actual
                        lista[j][1] = str(nuevo_stock)
                        vendidas += 1

                        codigo_negativo = "-" + codigo
                        encontrado = 0
                        k = 0
                        while k < len(lista):
                            if lista[k][0] == codigo_negativo:
                                total = int(lista[k][1])
                                lista[k][1] = str(total + unidades)
                                encontrado = 1
                            k += 1

                        if encontrado == 0:
                            nueva_fila = [codigo_negativo, str(unidades)]
                            lista.append(nueva_fila)

                        if lista[j][1] == "0":
                            nueva = []
                            k = 0
                            while k < len(productos_disponibles):
                                if productos_disponibles[k] != codigo:
                                    nueva.append(productos_disponibles[k])
                                k += 1
                            produtos_disponibles = nueva

                    j += 1
                else:
                    intentos += 1

    j = 0
    while j < len(lista):
        i = j + 1
        while i >= 0:
            if int(lista[i][0]) > int(lista[i + 1][0]):
                temp = lista[i]
                lista[i] = lista[i + 1]
                lista[i + 1] = temp
            i -= 1
        j += 1

    archivo = open(ruta, "w")
    i = 0
    while i < len(lista):
        fila = lista[i][0] + ": " + lista[i][1] + "\n"
        archivo.write(fila)
        i += 1
    archivo.close()


vender_productos("inventario.txt", 3)
