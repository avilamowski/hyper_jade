def validar_texto():
    valido = False
    while not valido:
        texto = input(
            "ingrese texto:"
        )  # como no espesifca si se frena o no el programa lo segui hasta que sea valido
        if texto != "":
            valido = True
        else:
            print(False)
    val = False
    i = 0
    while len(texto) and not val:
        if (
            ("0" <= texto[i] <= "9")
            or ("a" <= texto[i] <= "z")
            or ("A" <= texto[i] <= "Z")
            or texto == " "
        ):
            val = True
        i += 1
    if val:
        print("True")
    else:
        print("False")


validar_texto()

# lista=["hola","como","","estas","","0",] ejemplo q use suponiedno q anda split con \n


def mostrar(lista):
    pos = 1
    for i in range(len(lista)):
        if lista[i] != "":
            print(pos, ":", lista[i])
            pos += 1


mostrar(lista)


# en la consigna no se aclara sobre que es el valor de los numeros q se saca en el promedio, interpete q se referia a la suma de los numeros sobre cantidad de ellos
def promedio_numeros(texto):
    sum_num = 0
    cant_num = 0
    for i in texto:
        if i in "1234567890":
            sum_num += int(i)
            cant_num += 1
    try:
        promedio = sum_num / cant_num
        return promedio
    except:
        print("no hay numeros")


promedio = promedio_numeros(texto)
print("El promedio de los números en el texto es", promedio)


# lista=["a","a","e","0","ccc","b","dd","","b"] lista que use para itentar
def unicas_palabras(lista):
    unicas = []
    for i in range(len(lista)):
        if lista[i] not in "1234567890" and lista[i] != "":
            if lista[i] not in unicas:
                unicas.append(lista[i])
    return unicas


unicas = unicas_palabras(lista)
print(unicas)


def ordenar_custom(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if len(lista[i]) < len(lista[j]):
                lista[i], lista[j] = lista[j], lista[i]
    return lista


lista_ord = ordenar_custom(unicas)
print(lista_ord)


def contenido(lista_ord):
    contenido = ""
    for i in lista_ord:
        contenido += i + "\n"
    return contenido


def sin_repetir(contenido):
    try:
        with open("sin_repeticiones.txt", "w") as archivo:
            escritor = archivo.write(contenido)
            return escritor
    except:
        print("hubo un error para crear archivo")


contenido = contenido(lista_ord)
escrito = sin_repetir(contenido)
