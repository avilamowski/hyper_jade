def validar(texto):
    valid = True
    if len(texto) == 0:
        valid = False
    i = 0
    while i < len(texto):
        if not (
            "0" <= str(texto[i]) <= "9"
            or "a" <= texto[i] <= "z"
            or "A" <= texto[i] <= "Z"
            or texto[i] == " "
            or texto[i] == "\n"
        ):
            valid = False
        i += 1
    return valid


def mi_split(texto, separador):
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


def mi_strip(texto):
    inicio, fin = 0, len(texto) - 1
    while inicio <= fin and (texto[inicio] == " " or texto[inicio] == "\n"):
        inicio += 1
    while fin >= inicio and (texto[fin] == " " or texto[fin] == "\n"):
        fin -= 1
    return texto[inicio : fin + 1]


def texto_sin_espacio(texto):
    texto_bien = ""
    i = 0
    while i < len(texto) - 1:
        if not (texto[i] == " " and texto[i + 1] == " "):
            texto_bien += texto[i]
        i += 1
    texto_bien += texto[i]
    return texto_bien


########2########
def mostrar_lineas(texto):
    if validar(texto):
        lista = mi_split(texto_sin_espacio(texto), "\n")
        for i in range(len(lista)):
            if lista[i] != " " and lista[i] != "":
                print(f"{i+1}: {mi_strip(lista[i])}")
        print("\n")
    else:
        print("Texto inválido")


def lista_palabra(texto):
    if validar(texto):
        lista_palabra_bien = []
        lista_palabras = mi_split(texto, " ")
        for i in range(len(lista_palabras)):
            lista_palabra_bien.append(mi_strip(lista_palabras[i]))

        return lista_palabra_bien
    else:
        print("Texto invalido")


def extraer_num(lista_palabra_bien):
    numeros = []
    cont = 0
    for i in range(len(lista_palabra_bien)):
        try:
            if int(lista_palabra_bien[i]) / 2:
                numeros.append(int(lista_palabra_bien[i]))

        except:
            cont = 0
    return numeros


##########3##########
def promedio_numeros(texto):
    lista_palabra_bien = lista_palabra(texto)
    numeros = extraer_num(lista_palabra_bien)
    suma = 0
    cont = 0
    for i in range(len(numeros)):
        suma += numeros[i]
        cont += 1
    promedio = suma / cont

    print(f"El promedio de los numeros en el texto es {promedio}\n")


def ord_des(lista):
    long = len(lista)
    for i in range(long):
        for j in range(long - i - 1):
            if len(lista[j]) < len(lista[j + 1]):

                aux = lista[j]
                lista[j] = lista[j + 1]
                lista[j + 1] = aux


def crear_txt(FILE, sin_repe):
    try:
        with open(FILE, "w", encoding="utf-8") as archivo:
            for i in range(len(sin_repe)):
                archivo.write(f"{sin_repe[i]}\n")

    except FileNotFoundError:
        return None


##########4#############
def sin_repetir(texto):
    if validar(texto):
        sin_repe = []
        lista_palabra_bien = lista_palabra(texto)
        numeros = str(extraer_num(lista_palabra_bien))

        for i in range(len(lista_palabra_bien)):
            if lista_palabra_bien[i] not in sin_repe:
                if lista_palabra_bien[i] not in numeros:
                    sin_repe.append(lista_palabra_bien[i])

        ord_des(sin_repe)

        crear_txt("sin_repeticiones.txt", sin_repe)


texto = "Hola  gente del universo\n \n123 \n Chau todos\n 90 adios"


mostrar_lineas(texto)
promedio_numeros(texto)
sin_repetir(texto)
