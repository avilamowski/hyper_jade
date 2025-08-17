import random


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
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


# -------------------------------------------------------------------------------------------------
def vender_productos(ruta, cantidad):
    if cantidad <= 0:
        raise Exception("La cantidad debe ser positiva.")
    elif (
        cantidad / cantidad
    ) == 1.0:  # si ya se que es mayor a cero, verifico que no sea float
        raise Exception("Debe ser INT.")

    cantidades = []
    codigos = []

    archivo_existe = True
    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            linea = archivo.readline()
            while linea != "":
                partes = split_custom(strip_custom(linea), ":")
                if len(partes) == 2:
                    codigo = int(partes[0])
                    cantidad = int(partes[1])
                    codigos.append(codigo)
                    cantidades.append(cantidad)
                linea = archivo.readline()
    except FileNotFoundError:
        archivo_existe = False

    # Si no existe, lo creo
    if not archivo_existe:
        with open(ruta, "w", encoding="utf-8") as archivo_nuevo:
            archivo_nuevo.write("1000: 100" + "\n")
            archivo_nuevo.write("1001: 100" + "\n")
            archivo_nuevo.write("1002: 100" + "\n")
            archivo_nuevo.write("1003: 100" + "\n")
            archivo_nuevo.write("1004: 100" + "\n")
            archivo_nuevo.write("1005: 100" + "\n")
            archivo_nuevo.write("1006: 100" + "\n")
            archivo_nuevo.write("1007: 100" + "\n")
            archivo_nuevo.write("1008: 100" + "\n")
            archivo_nuevo.write("1009: 100" + "\n")

    cod_pos = []  # inventario
    stock_pos = []
    cod_neg = []  # ventas historicas
    stock_neg = []
    i = 0
    while i < len(codigos):
        if codigos[i] > 0:
            cod_pos.append(codigos[i])
            stock_pos.append(cantidades[i])
        else:
            cod_neg.append(codigos[i])
            stock_neg.append(cantidades[i])
        i += 1

    # realizo las ventas
    codigo_vendido = []
    cuanto = []
    k = 0
    while k < cantidad:
        n = random.randint(1000, 1009)
        x = random.randint(1, 5)
        codigo_vendido.append(n)
        cuanto.append(x)
        k += 1

    i = 0
    while i < len(codigo_vendido):
        j = 0
        while j < len(cod_pos):
            if codigo_vendido[i] == cod_pos[j]:  # si el codigo es igual
                if (
                    stock_pos[j] < cuanto[i]
                ):  # si es menor a lo que se vendio, ya le pongo cero
                    stock_pos[j] = 0
                elif stock_pos[j] > cuanto[i]:  # caso contrario, se lo resto
                    stock_pos[j] -= cuanto[i]

            if codigo_vendido[i] == (cod_neg[j][1:]):
                stock_neg[j] += cuanto[i]  # cod_neg era ventas historicas

            j += 1

        i += 1

    # escribo todo nuevamente como me quedo
    with open(ruta, "w", encoding="utf-8") as archivo:
        k = 0
        while k < len(cod_pos):
            archivo.write(str(cod_pos[k]) + ":" + str(stock_pos[k]) + "\n")
            k += 1

        h = 0
        while h < len(cod_neg):
            archivo.write(str(cod_neg[h]) + ":" + str(stock_neg[h]) + "\n")
            h += 1


# PROGRAMA PRINCIPAL
cantidad = 5
print(vender_productos("inventario.txt", cantidad))


cantidad = 1.0
if type(cantidad) != int:
    print("False")
