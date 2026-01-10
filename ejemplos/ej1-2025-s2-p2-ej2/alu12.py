import csv


def leer_archivo(nombre1):
    nombre_i = []
    precio_i = []
    cantidad_i = []
    try:
        with open(nombre1, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                nombre_i.append(fila[0])
                precio_i.append(fila[1])
                cantidad_i.append(fila[2])
    except FileNotFoundError:
        print("El archivo no existe")
    except Exception as e:
        print(f"Ocurrió un error {e}")
    return nombre_i, precio_i, cantidad_i


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


def strip_custom(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " " or texto[fin] == "\n":
        fin -= 1
    return texto[inicio : fin + 1]


def leer_archivo_de_texto(nombre2):
    nombre_v = []
    cantidad_v = []
    try:
        with open(nombre2, "r", encoding="utf-8") as archivo:
            for fila in archivo:
                contenido = split_custom(fila, ":")
                nombre_v.append(contenido[0])
                cantidad_v.append(strip_custom(contenido[1]))
    except FileNotFoundError:
        print("El archivo no existe")
    except Exception as e:
        print(f"Ocurrió un error {e}")
    return nombre_v, cantidad_v


def lista_sin_repetir(cantidad_v, nombre_v):
    nueva_c_v = []
    nueva_n_v = []
    for i in range(len(cantidad_v)):
        if nombre_v[i] not in nueva_n_v:
            nueva_n_v.append(nombre_v[i])
            nueva_c_v.append(cantidad_v[i])
        else:
            for j in range(len(nueva_n_v)):
                if nueva_n_v[j] == nombre_v[i]:
                    aux = int(nueva_c_v[j])
                    aux2 = int(cantidad_v[i])
                    nueva_c_v[j] = aux + aux2
    return nueva_c_v, nueva_n_v


def costos(nueva_c_v, nueva_n_v, precio_i, nombre_i, cantidad_i):
    costos_productos = 0
    for i in range(len(nueva_n_v)):
        if nueva_n_v[i] in nombre_i[i]:
            if float(nueva_c_v[i]) > float(cantidad_i[i]):
                c = float(cantidad_i[i]) * float(precio_i[i])
                costos_productos += c
            else:
                c = float(nueva_c_v[i]) * float(precio_i[i])
                costos_productos += c
    return costos_productos


def main():
    nombre1 = "inventario.csv"
    nombre2 = "venta.txt"
    nombre_i, precio_i, cantidad_i = leer_archivo(nombre1)
    nombre_v, cantidad_v = leer_archivo_de_texto(nombre2)
    nueva_c_v, nueva_n_v = lista_sin_repetir(cantidad_v, nombre_v)
    costos_productos = costos(nueva_c_v, nueva_n_v, precio_i, nombre_i, cantidad_i)


main()
