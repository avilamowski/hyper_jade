# FALTA UNIFICAR TODAS LAS FUNCIONES, HICE FUNCION POR FUNCION POR CADA PTO
# PERDONNNNNNN


# 1
def validacion_caracter(cadena):
    caracteres_valido = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n"
    )
    for s in cadena:
        if s not in caracteres_valido:
            return False
    return True


def validar(texto):
    valido = validacion_caracter(texto)
    return valido


# 2
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


def boorar_vac(lista):
    nueva = []
    for elementos in lista:
        if elementos != "":
            nueva.append([strip_custom(elementos), " "])
    return nueva


def mostrar(lista):
    for elementos in range(len(lista)):
        print(
            elementos + 1,
            ":",
            lista[elementos][0],
        )


def mostrar_lineas(texto):
    lista = split_custom(cad, separador="\n")
    lista = boorar_vac(lista)
    mostrar(lista)


# 3
def num(texto):
    es_num = False
    num = ""
    numeros = []
    validos = "0123456789"
    for caracteres in texto:
        print(caracteres)
        if caracteres in validos:
            num += caracteres
            es_num = True
        elif caracteres not in validos and es_num:
            numeros.append(int(num))
            num = ""
    return numeros


def suma_prom(lista):
    suma = 0
    for elementos in lista:
        suma += elementos
    if len(lista) > 0:

        promedio = suma / len(lista)
    else:
        promedio = 0
    print("El promedio de los números en el texto es", promedio)


def promedio_numeros(texto):
    numeros = num(texto)
    suma_prom(numeros)


# PARTE DEL 4


def crear(texto):
    texto = ""
    with open("sin_repeticiones.txt", "w", encoding="utf-8") as f:
        for linea in a:
            texto += f"{linea[0]}\n"
        f.write(texto)
