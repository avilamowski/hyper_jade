from random import randint


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


def comprobar_archivo(ruta_archivo):

    ids = []
    cantidades = []

    try:
        with open(ruta_archivo, "r", encoding="UTF-8") as archivo:
            # entiendo que no tiene encabezado, sino pondría archivo.readline()
            for linea in archivo:
                linea = strip_custom(linea)
                partes = split_custom(linea, ":")

                if len(partes) == 2:
                    ids.append(strip_custom(partes[0]))
                    cantidades(strip_custom(partes[1]))

                else:
                    print("Error: linea no incluida por no tener ambos datos")

    except FileNotFoundError:
        print("Error: No se encontró el archivo")
        return False

    return ids, cantidades


def crear_lista_ids_unicos(lista_ids):

    ids_unicos = []

    for id in lista_ids:
        if id not in ids_unicos:
            ids_unicos.append(id)

    lista_ids_orginales = []

    for elem in ids_unicos:
        if elem[0] != "-":  # busco los que son ventas para no incluirlos
            if elem not in lista_ids_orginales:
                lista_ids_orginales.append(elem)

    return lista_ids_orginales


def validar_cantidad(cantidad):
    try:
        num = int(cantidad)
        if 0 < num:
            return True

    except ValueError:
        print("El valor a ingresar debe ser un numero entero positivo")
        return False


def vender_productos(ruta, cantidad):

    ids_chequeados, cantidades_chequeadas = comprobar_archivo(ruta)

    cantidad_validada = True
    while cantidad_validada:
        if validar_cantidad(cantidad) == True:
            cantidad_validada = False
        else:
            print("Por favor, vuelva a ingresar la cantidad del producto a vender:")
            cantidad = input("Nueva cantidad: ")

    if comprobar_archivo(ruta) == False:

        ids_inventados = []
        cantidades_inventadas = []

        with open(ruta, "w", encoding="UTF-8") as archivo:
            i = 0
            c = 1000
            while i < 10:
                ids_inventados.append(c)
                cantidades_inventadas(100)
                c += 1
                i += 1

            for j in range(len(ids_inventados)):
                archivo.write(ids_inventados[j], ": ", cantidades_inventadas[i])

            p = 0
            while p < cantidad:
                g = random.randint(0, len(ids_inventados) - 1)
                h = random.randint(1, 5)

                for i in range(len(ids_inventados)):
                    if ids_inventados[i] == ids_inventados[g]:
                        if cantidades_inventadas[i] < h:
                            cantidades_inventadas[i] = 0
                            archivo.write(
                                "-", ids_inventados[g], ": ", cantidades_inventadas[i]
                            )
                        else:
                            cantidades_inventadas[i] -= h
                            archivo.write("-", ids_inventados[g], ": ", h)
                p += 1

    else:

        lista_ids_unicos = crear_lista_ids_unicos(ids_chequeados)
        with open(ruta, "a", encoding="UTF-8") as archivo:

            p = 0
            while p < cantidad:
                g = random.randint(0, len(lista_ids_unicos) - 1)
                h = random.randint(1, 5)

                for i in range(len(ids_chequeados)):
                    if ids_chequeados[i] == lista_ids_unicos[g]:
                        if cantidades_chequeadas[i] < h:
                            cantidades_chequeadas[i] = 0
                            archivo.write(
                                "-", lista_ids_unicos[g], ": ", cantidades_chequeadas[i]
                            )
                        else:
                            cantidades_chequeadas[i] -= h
                            archivo.write("-", lista_ids_unicos[g], ": ", h)

                p += 1
