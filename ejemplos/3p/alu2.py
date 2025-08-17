# Funcion extra
def strip_custom(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and texto[inicio] == " ":
        inicio += 1
    while fin >= inicio and texto[fin] == " ":
        fin -= 1
    return texto[inicio : fin + 1]


# Funcion extra
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


# Leo/genero archivo en caso de que este no exista
def read_file(namefile):
    codigo_p = []
    stock = []

    try:
        with open(namefile, "r", encoding="utf-8") as fd:

            for line in fd:
                line = line[:-1]
                lista = split_custom(line, ":")
                codigo_p.append(int(lista[0]))
                stock.append(int(strip_custom(lista[1])))

    except FileNotFoundError:
        with open(namefile, "w", encoding="utf-8") as fd:
            for codigo in range(1000, 1010):
                fd.write(f"{codigo}: 100\n")
                codigo_p.append(codigo)
                stock.append(100)

    return codigo_p, stock


# Verifico que la cantidad ingresada sea valida
def verifico_cantidad(cantidad):

    try:
        cantidad = int(cantidad)
        if cantidad <= 0:
            raise ValueError

        return cantidad

    except Exception as e:
        print("El ingreso no es valido")


# Actualizo el archivo
def write_new_file(ruta, codigo_p, stock):
    with open(ruta, "w", encoding="utf-8") as fd:
        for pos in range(len(codigo_p)):
            fd.write(f"{codigo_p[pos]}: {stock[pos]}\n")


def vender_productos(ruta, cantidad):

    cantidad = verifico_cantidad(cantidad)
    codigo_p, stock = read_file(ruta)

    i = 1
    while i <= cantidad:

        producto = random.randint(1000, 1009)
        venta = random.randint(1, 5)

        for p in range(len(codigo_p)):

            # si el codigo es positivo
            if codigo_p[p] == producto:

                if venta <= stock[p]:
                    stock[p] = stock[p] - venta

                else:
                    stock[p] = 0

            # si el codigo es negativo
            elif codigo_p[p] == (-producto):
                stock[p] += venta

            if -producto not in codigo_p:
                codigo_p.append(-producto)
                stock.append(venta)

        i += 1

    write_new_file(ruta, codigo_p, stock)


def main():
    vender_productos("inventario.txt", 3)


main()
